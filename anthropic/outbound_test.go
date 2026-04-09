package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestToMessagesRequest_BasicTextExchange(t *testing.T) {
	f := false
	req := &api.ChatRequest{
		Model: "anthropic/claude-3.5-haiku",
		Messages: []api.Message{
			{Role: "system", Content: "You are concise."},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
			{Role: "user", Content: "Bye"},
		},
		Stream:  &f,
		Options: map[string]any{"temperature": 0.5, "num_predict": 200},
	}

	out, err := ToMessagesRequest(req)
	if err != nil {
		t.Fatalf("ToMessagesRequest: %v", err)
	}
	if out.Model != "anthropic/claude-3.5-haiku" {
		t.Errorf("model: got %q", out.Model)
	}
	if out.System != "You are concise." {
		t.Errorf("system: got %q", out.System)
	}
	if out.MaxTokens != 200 {
		t.Errorf("max_tokens: got %d, want 200", out.MaxTokens)
	}
	if out.Temperature == nil || *out.Temperature != 0.5 {
		t.Errorf("temperature: got %v", out.Temperature)
	}
	if out.Stream {
		t.Errorf("stream should be false")
	}
	if len(out.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "user" || out.Messages[1].Role != "assistant" || out.Messages[2].Role != "user" {
		t.Errorf("roles: %v %v %v", out.Messages[0].Role, out.Messages[1].Role, out.Messages[2].Role)
	}
	if len(out.Messages[0].Content) != 1 || out.Messages[0].Content[0].Type != "text" {
		t.Errorf("first user content wrong: %+v", out.Messages[0].Content)
	}
	if out.Messages[0].Content[0].Text == nil || *out.Messages[0].Content[0].Text != "Hello" {
		t.Errorf("first user text wrong")
	}
}

func TestToMessagesRequest_DefaultMaxTokens(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model:    "x",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.MaxTokens != defaultOutboundMaxTokens {
		t.Errorf("want default %d, got %d", defaultOutboundMaxTokens, out.MaxTokens)
	}
}

func TestToMessagesRequest_MergesSystemMessages(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model: "x",
		Messages: []api.Message{
			{Role: "system", Content: "A"},
			{Role: "system", Content: "B"},
			{Role: "user", Content: "hi"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.System != "A\n\nB" {
		t.Errorf("system join: got %q", out.System)
	}
}

func TestToMessagesRequest_MergesConsecutiveSameRole(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model: "x",
		Messages: []api.Message{
			{Role: "user", Content: "first"},
			{Role: "user", Content: "second"},
			{Role: "assistant", Content: "ack"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 2 {
		t.Fatalf("want 2 merged, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "user" {
		t.Errorf("first role: %v", out.Messages[0].Role)
	}
	if len(out.Messages[0].Content) != 2 {
		t.Errorf("merged content blocks: %d", len(out.Messages[0].Content))
	}
}

func TestToMessagesRequest_ToolRoleBecomesUserToolResult(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model: "x",
		Messages: []api.Message{
			{Role: "user", Content: "do a thing"},
			{Role: "assistant", ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.NewToolCallFunctionArguments()}},
			}},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 3 {
		t.Fatalf("want 3 msgs, got %d", len(out.Messages))
	}
	if out.Messages[1].Role != "assistant" {
		t.Errorf("assistant msg role: %v", out.Messages[1].Role)
	}
	if out.Messages[1].Content[0].Type != "tool_use" || out.Messages[1].Content[0].ID != "call_1" {
		t.Errorf("tool_use block: %+v", out.Messages[1].Content[0])
	}
	if out.Messages[2].Role != "user" {
		t.Errorf("tool result should be user role: %v", out.Messages[2].Role)
	}
	if out.Messages[2].Content[0].Type != "tool_result" || out.Messages[2].Content[0].ToolUseID != "call_1" {
		t.Errorf("tool_result block: %+v", out.Messages[2].Content[0])
	}
}

func TestToMessagesRequest_ThinkingEnabled(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model:    "x",
		Messages: []api.Message{{Role: "user", Content: "think"}},
		Think:    &api.ThinkValue{Value: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Thinking == nil || out.Thinking.Type != "enabled" {
		t.Errorf("thinking config: %+v", out.Thinking)
	}
}

func TestToMessagesRequest_ToolsConverted(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model:    "x",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Tools: api.Tools{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get current weather",
					Parameters: api.ToolFunctionParameters{
						Type:     "object",
						Required: []string{"location"},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 {
		t.Fatalf("tools: %d", len(out.Tools))
	}
	if out.Tools[0].Name != "get_weather" || out.Tools[0].Description != "Get current weather" {
		t.Errorf("tool: %+v", out.Tools[0])
	}
	// InputSchema should be valid JSON with "object" type.
	var sc map[string]any
	if err := json.Unmarshal(out.Tools[0].InputSchema, &sc); err != nil {
		t.Fatalf("input_schema unmarshal: %v", err)
	}
	if sc["type"] != "object" {
		t.Errorf("schema type: %v", sc["type"])
	}
}

func TestToMessagesRequest_StopSequences(t *testing.T) {
	out, err := ToMessagesRequest(&api.ChatRequest{
		Model:    "x",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"stop": []any{"END", "STOP"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.StopSequences) != 2 || out.StopSequences[0] != "END" || out.StopSequences[1] != "STOP" {
		t.Errorf("stop: %v", out.StopSequences)
	}
}
