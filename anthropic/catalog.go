package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// catalogTTL is how long a successfully-fetched catalog is considered fresh.
// KiloCode's model list changes on the order of days, so a short TTL is
// wasteful; a long TTL masks new model availability. 10 minutes is a pragmatic
// middle ground for interactive use.
const catalogTTL = 10 * time.Minute

// ModelInfo is a subset of the KiloCode/OpenRouter model record containing
// only the fields Ollama needs to synthesize accurate show/list responses.
type ModelInfo struct {
	ID                  string
	Name                string
	Description         string
	ContextLength       int
	MaxCompletionTokens int
	InputModalities     []string
	SupportedParameters []string
}

// HasTool reports whether the model advertises function/tool calling.
func (m *ModelInfo) HasTool() bool {
	return hasParam(m.SupportedParameters, "tools")
}

// HasReasoning reports whether the model advertises Anthropic-style
// thinking/reasoning. When true, passing ThinkingConfig to the upstream is
// safe; when false, it triggers a 400 from the gateway.
func (m *ModelInfo) HasReasoning() bool {
	return hasParam(m.SupportedParameters, "reasoning") ||
		hasParam(m.SupportedParameters, "include_reasoning")
}

// HasVision reports whether the model accepts image input modality.
func (m *ModelInfo) HasVision() bool {
	for _, mod := range m.InputModalities {
		if strings.EqualFold(mod, "image") {
			return true
		}
	}
	return false
}

func hasParam(params []string, want string) bool {
	for _, p := range params {
		if strings.EqualFold(p, want) {
			return true
		}
	}
	return false
}

// rawModelRecord is the wire shape returned by /api/openrouter/models. Only
// fields we consume are declared; json.Unmarshal discards the rest.
type rawModelRecord struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Architecture struct {
		InputModalities []string `json:"input_modalities"`
	} `json:"architecture"`
	TopProvider struct {
		ContextLength       int `json:"context_length"`
		MaxCompletionTokens int `json:"max_completion_tokens"`
	} `json:"top_provider"`
	ContextLength       int      `json:"context_length"`
	SupportedParameters []string `json:"supported_parameters"`
}

type catalogResponse struct {
	Data []rawModelRecord `json:"data"`
}

// Catalog is a process-wide lazy cache of the KiloCode model list. A single
// instance is shared via DefaultCatalog; tests can construct their own to
// inject mock HTTP responses.
type Catalog struct {
	BaseURL    string
	HTTPClient *http.Client
	APIKey     string

	mu        sync.RWMutex
	fetchedAt time.Time
	models    map[string]*ModelInfo
}

// DefaultCatalog is the process-wide catalog used by server code. Callers
// should read from it via Lookup; it fetches on first use.
var DefaultCatalog = &Catalog{
	BaseURL:    kilocodeDefaultBaseURL,
	HTTPClient: &http.Client{Timeout: 30 * time.Second},
}

// Lookup returns the ModelInfo for the given model id, fetching the catalog
// on first call or when the cache has expired. A nil return with nil error
// means the catalog fetched successfully but did not contain that id.
func (c *Catalog) Lookup(ctx context.Context, id string) (*ModelInfo, error) {
	c.mu.RLock()
	fresh := time.Since(c.fetchedAt) < catalogTTL && c.models != nil
	if fresh {
		m := c.models[id]
		c.mu.RUnlock()
		return m, nil
	}
	c.mu.RUnlock()

	if err := c.refresh(ctx); err != nil {
		return nil, err
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.models[id], nil
}

func (c *Catalog) refresh(ctx context.Context) error {
	key := c.APIKey
	if key == "" {
		k, err := loadKiloCodeKey()
		if err != nil {
			return err
		}
		key = k
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Accept", "application/json")

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("catalog fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("catalog fetch: status %s", resp.Status)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return fmt.Errorf("catalog read: %w", err)
	}

	var parsed catalogResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return fmt.Errorf("catalog decode: %w", err)
	}

	models := make(map[string]*ModelInfo, len(parsed.Data))
	for i := range parsed.Data {
		r := &parsed.Data[i]
		ctxLen := r.ContextLength
		if ctxLen == 0 {
			ctxLen = r.TopProvider.ContextLength
		}
		models[r.ID] = &ModelInfo{
			ID:                  r.ID,
			Name:                r.Name,
			Description:         r.Description,
			ContextLength:       ctxLen,
			MaxCompletionTokens: r.TopProvider.MaxCompletionTokens,
			InputModalities:     r.Architecture.InputModalities,
			SupportedParameters: r.SupportedParameters,
		}
	}

	c.mu.Lock()
	c.models = models
	c.fetchedAt = time.Now()
	c.mu.Unlock()
	return nil
}

// All returns a snapshot of all cached models. If the cache is empty it
// triggers a fetch first. Used for list-style responses.
func (c *Catalog) All(ctx context.Context) ([]*ModelInfo, error) {
	c.mu.RLock()
	fresh := time.Since(c.fetchedAt) < catalogTTL && c.models != nil
	c.mu.RUnlock()
	if !fresh {
		if err := c.refresh(ctx); err != nil {
			return nil, err
		}
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]*ModelInfo, 0, len(c.models))
	for _, m := range c.models {
		out = append(out, m)
	}
	return out, nil
}
