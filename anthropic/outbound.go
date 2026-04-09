package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// defaultOutboundMaxTokens is used when the incoming api.ChatRequest has no
// num_predict option. Anthropic Messages API requires max_tokens to be set.
const defaultOutboundMaxTokens = 4096

// ToMessagesRequest converts an Ollama api.ChatRequest into an Anthropic
// MessagesRequest. This is the inverse of FromMessagesRequest and is used when
// Ollama acts as an outbound client to an Anthropic-compatible upstream such
// as Anthropic, OpenRouter, or KiloCode Gateway.
//
// Mapping rules:
//   - Leading system messages are concatenated and moved into MessagesRequest.System.
//   - Consecutive messages of the same role are merged into a single MessageParam.
//   - Role "tool" becomes a user-role MessageParam containing tool_result blocks.
//   - Image bytes (api.ImageData) are emitted as base64 image blocks.
//   - api.Tool (function type) maps to Anthropic Tool with input_schema derived
//     from ToolFunction.Parameters.
//   - Options num_predict/temperature/top_p/top_k/stop map to the corresponding
//     Anthropic request fields. Max tokens defaults to defaultOutboundMaxTokens.
//   - Think != nil emits ThinkingConfig{Type: "enabled"}.
func ToMessagesRequest(r *api.ChatRequest) (*MessagesRequest, error) {
	if r == nil {
		return nil, fmt.Errorf("nil chat request")
	}

	out := &MessagesRequest{
		Model:     r.Model,
		MaxTokens: defaultOutboundMaxTokens,
	}
	if r.Stream != nil {
		out.Stream = *r.Stream
	}

	// Options → top-level request fields. Anthropic has no generic "options"
	// bag; parameters live on MessagesRequest itself.
	if r.Options != nil {
		if v, ok := r.Options["num_predict"]; ok {
			if n, ok := asInt(v); ok && n > 0 {
				out.MaxTokens = n
			}
		}
		if v, ok := r.Options["temperature"]; ok {
			if f, ok := asFloat(v); ok {
				out.Temperature = &f
			}
		}
		if v, ok := r.Options["top_p"]; ok {
			if f, ok := asFloat(v); ok {
				out.TopP = &f
			}
		}
		if v, ok := r.Options["top_k"]; ok {
			if n, ok := asInt(v); ok {
				out.TopK = &n
			}
		}
		if v, ok := r.Options["stop"]; ok {
			out.StopSequences = asStringSlice(v)
		}
	}

	if r.Think != nil && r.Think.Bool() {
		out.Thinking = &ThinkingConfig{Type: "enabled"}
	}

	// Convert tools. Ollama's api.Tool wraps a function definition with a
	// structured parameters schema; Anthropic expects input_schema as raw JSON.
	for _, t := range r.Tools {
		if !strings.EqualFold(t.Type, "function") && t.Type != "" {
			// Skip non-function tools; Anthropic only has custom function tools
			// plus its built-in server tools (web_search etc) which we don't
			// synthesize from this direction.
			continue
		}
		schema, err := json.Marshal(t.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("marshal tool %q parameters: %w", t.Function.Name, err)
		}
		out.Tools = append(out.Tools, Tool{
			Type:        "custom",
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: schema,
		})
	}

	// Walk messages: pull system, then build MessageParam blocks.
	var systemParts []string
	for i := 0; i < len(r.Messages); i++ {
		m := r.Messages[i]
		if m.Role == "system" {
			if m.Content != "" {
				systemParts = append(systemParts, m.Content)
			}
			continue
		}

		// Once we hit a non-system message, stop collecting system prompts.
		// Any later system message is folded back into text on its own
		// (Anthropic only has one top-level system field).
		remaining := r.Messages[i:]
		msgs, err := messagesToMessageParams(remaining)
		if err != nil {
			return nil, err
		}
		out.Messages = msgs
		break
	}

	if len(systemParts) > 0 {
		out.System = strings.Join(systemParts, "\n\n")
	}

	return out, nil
}

// messagesToMessageParams converts a slice of api.Message (with no leading
// system messages) into Anthropic MessageParam blocks. Consecutive messages of
// the same effective role are merged so the Anthropic conversation alternates
// user/assistant as required.
func messagesToMessageParams(msgs []api.Message) ([]MessageParam, error) {
	var out []MessageParam

	for _, m := range msgs {
		blocks, role, err := messageToBlocks(m)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}

		if len(out) > 0 && out[len(out)-1].Role == role {
			out[len(out)-1].Content = append(out[len(out)-1].Content, blocks...)
			continue
		}
		out = append(out, MessageParam{Role: role, Content: blocks})
	}

	return out, nil
}

// messageToBlocks converts a single api.Message into its Anthropic
// ContentBlock list plus the effective MessageParam role. Tool-role messages
// are re-framed as user messages carrying tool_result blocks, matching the
// Anthropic convention.
func messageToBlocks(m api.Message) ([]ContentBlock, string, error) {
	var blocks []ContentBlock

	switch strings.ToLower(m.Role) {
	case "tool":
		// Tool results must be attached to a user message as tool_result blocks.
		content := m.Content
		blocks = append(blocks, ContentBlock{
			Type:      "tool_result",
			ToolUseID: m.ToolCallID,
			Content:   content,
		})
		return blocks, "user", nil

	case "assistant":
		if m.Thinking != "" {
			t := m.Thinking
			blocks = append(blocks, ContentBlock{
				Type:     "thinking",
				Thinking: &t,
			})
		}
		if m.Content != "" {
			c := m.Content
			blocks = append(blocks, ContentBlock{
				Type: "text",
				Text: &c,
			})
		}
		for _, tc := range m.ToolCalls {
			blocks = append(blocks, ContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: tc.Function.Arguments,
			})
		}
		return blocks, "assistant", nil

	default: // "user" or anything else
		if m.Content != "" {
			c := m.Content
			blocks = append(blocks, ContentBlock{
				Type: "text",
				Text: &c,
			})
		}
		for _, img := range m.Images {
			blocks = append(blocks, ContentBlock{
				Type: "image",
				Source: &ImageSource{
					Type:      "base64",
					MediaType: detectImageMediaType(img),
					Data:      base64.StdEncoding.EncodeToString(img),
				},
			})
		}
		return blocks, "user", nil
	}
}

// detectImageMediaType sniffs the first few bytes of image data to pick a
// plausible MIME type. Anthropic accepts png/jpeg/gif/webp; we default to
// png when unsure since it's the most permissive for decoders.
func detectImageMediaType(b []byte) string {
	switch {
	case len(b) >= 8 && string(b[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(b) >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff:
		return "image/jpeg"
	case len(b) >= 6 && (string(b[:6]) == "GIF87a" || string(b[:6]) == "GIF89a"):
		return "image/gif"
	case len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WEBP":
		return "image/webp"
	default:
		return "image/png"
	}
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
		if f, err := n.Float64(); err == nil {
			return int(f), true
		}
	}
	return 0, false
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}

func asStringSlice(v any) []string {
	switch s := v.(type) {
	case []string:
		return s
	case string:
		if s == "" {
			return nil
		}
		return []string{s}
	case []any:
		out := make([]string, 0, len(s))
		for _, item := range s {
			if str, ok := item.(string); ok {
				out = append(out, str)
			}
		}
		return out
	}
	return nil
}
