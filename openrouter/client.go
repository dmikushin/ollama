// Package openrouter provides an HTTP client for OpenRouter's OpenAI-compatible API.
package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

const (
	defaultBaseURL = "https://openrouter.ai/api/v1"
	keyEnv         = "OPENROUTER_API_KEY"
)

// Client is an HTTP client for OpenRouter's OpenAI-compatible API.
type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

// NewClient constructs a Client, loading the API key from the OPENROUTER_API_KEY
// environment variable. Returns an error if the key is not set.
func NewClient() (*Client, error) {
	key := strings.TrimSpace(os.Getenv(keyEnv))
	if key == "" {
		return nil, fmt.Errorf("no OpenRouter API key: set %s", keyEnv)
	}
	return &Client{
		BaseURL: defaultBaseURL,
		APIKey:  key,
		HTTPClient: &http.Client{Timeout: 0},
	}, nil
}

// ClientFromEnv is equivalent to NewClient but accepts an optional base URL override
// via the OLLAMA_OPENROUTER_BASE_URL environment variable for testing.
func ClientFromEnv() (*Client, error) {
	c, err := NewClient()
	if err != nil {
		return nil, err
	}
	if override := strings.TrimSpace(os.Getenv("OLLAMA_OPENROUTER_BASE_URL")); override != "" {
		c.BaseURL = override
	}
	return c, nil
}

// Chat sends a ChatCompletionRequest to OpenRouter and returns the HTTP response.
// Callers are responsible for closing resp.Body.
func (c *Client) Chat(ctx context.Context, req *openai.ChatCompletionRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	// OpenRouter recommends these headers for better routing
	if os.Getenv("HTTP_REFERER") != "" {
		httpReq.Header.Set("Referer", os.Getenv("HTTP_REFERER"))
	}
	if os.Getenv("HTTP_X_TITLE") != "" {
		httpReq.Header.Set("X-Title", os.Getenv("HTTP_X_TITLE"))
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openrouter request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, readUpstreamError(resp)
	}

	return resp, nil
}

func readUpstreamError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024))

	var errResp openai.ErrorResponse
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
