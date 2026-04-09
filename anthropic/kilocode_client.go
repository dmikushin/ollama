package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
)

// KiloCode gateway defaults. The gateway exposes an OpenRouter-compatible
// endpoint that speaks Anthropic's Messages API at /v1/messages.
const (
	kilocodeDefaultBaseURL = "https://api.kilo.ai/api/openrouter"
	kilocodeKeyEnv         = "KILOCODE_API_KEY"
	kilocodeKeyFile        = ".kilocode/key"
)

// OutboundClient is an HTTP client for an Anthropic-compatible upstream. It
// POSTs MessagesRequest JSON to BaseURL + /v1/messages and returns the raw
// response body so callers can decode either a single JSON response or a
// streaming SSE event stream.
type OutboundClient struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	// UserAgent, if non-empty, is sent as the User-Agent header.
	UserAgent string
}

// NewKiloCodeClient constructs an OutboundClient pointed at the KiloCode
// gateway, loading the bearer key from ~/.kilocode/key (falling back to the
// KILOCODE_API_KEY environment variable). Returns an error if no key source
// is available.
func NewKiloCodeClient() (*OutboundClient, error) {
	key, err := loadKiloCodeKey()
	if err != nil {
		return nil, err
	}
	return &OutboundClient{
		BaseURL: kilocodeDefaultBaseURL,
		APIKey:  key,
		HTTPClient: &http.Client{
			// Do not enforce a total timeout: streaming responses may be long-lived.
			// Connect/TLS/headers timeouts are handled by the default transport.
			Timeout: 0,
		},
		UserAgent: "ollama-kilocode/1.0",
	}, nil
}

func loadKiloCodeKey() (string, error) {
	if key := strings.TrimSpace(os.Getenv(kilocodeKeyEnv)); key != "" {
		return key, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locate home dir: %w", err)
	}
	path := filepath.Join(home, kilocodeKeyFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("no KiloCode API key: set %s or create %s", kilocodeKeyEnv, path)
		}
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	key := strings.TrimSpace(string(data))
	if key == "" {
		return "", fmt.Errorf("%s is empty", path)
	}
	return key, nil
}

// Messages POSTs a MessagesRequest to the upstream and returns the HTTP
// response. Callers are responsible for closing resp.Body. Non-2xx status
// codes are returned as an error with an api.StatusError wrapping the
// upstream error body, and resp.Body is closed before returning.
func (c *OutboundClient) Messages(ctx context.Context, req *MessagesRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal messages request: %w", err)
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/v1/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	if c.UserAgent != "" {
		httpReq.Header.Set("User-Agent", c.UserAgent)
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("kilocode request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, readUpstreamError(resp)
	}

	return resp, nil
}

// readUpstreamError consumes an error response body and returns an
// api.StatusError so existing handler plumbing can surface it to the client.
func readUpstreamError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024))

	// Attempt to decode as an Anthropic ErrorResponse for a cleaner message.
	var errResp ErrorResponse
	if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
		return api.StatusError{
			StatusCode:   resp.StatusCode,
			Status:       resp.Status,
			ErrorMessage: errResp.Error.Message,
		}
	}

	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = resp.Status
	}
	return api.StatusError{
		StatusCode:   resp.StatusCode,
		Status:       resp.Status,
		ErrorMessage: msg,
	}
}

// KiloCodeClientFromEnv is a convenience constructor that is equivalent to
// NewKiloCodeClient but also accepts an override base URL via the environment
// variable OLLAMA_KILOCODE_BASE_URL for testing against a local mock.
func KiloCodeClientFromEnv() (*OutboundClient, error) {
	c, err := NewKiloCodeClient()
	if err != nil {
		return nil, err
	}
	if override := strings.TrimSpace(os.Getenv("OLLAMA_KILOCODE_BASE_URL")); override != "" {
		c.BaseURL = override
	}
	return c, nil
}
