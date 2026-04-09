package anthropic

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

// collect runs the converter against sseBody and returns every ChatResponse
// emitted via the callback plus any terminal error from Run.
func collect(t *testing.T, sseBody string, parser parsers.Parser) ([]api.ChatResponse, error) {
	t.Helper()
	var out []api.ChatResponse
	conv := NewInboundStreamConverter("test/model", func(r api.ChatResponse) error {
		out = append(out, r)
		return nil
	})
	conv.TextParser = parser
	err := conv.Run(strings.NewReader(sseBody))
	return out, err
}

func TestInboundStream_BasicText(t *testing.T) {
	body := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-3","content":[],"usage":{"input_tokens":7,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":7,"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`
	chunks, err := collect(t, body, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks (2 text + 1 final), got %d", len(chunks))
	}
	if chunks[0].Message.Content != "Hello" || chunks[1].Message.Content != ", world" {
		t.Errorf("text deltas: %q, %q", chunks[0].Message.Content, chunks[1].Message.Content)
	}
	final := chunks[2]
	if !final.Done {
		t.Error("final chunk not Done")
	}
	if final.DoneReason != "stop" {
		t.Errorf("done reason: %q", final.DoneReason)
	}
	if final.Metrics.PromptEvalCount != 7 || final.Metrics.EvalCount != 3 {
		t.Errorf("metrics: %+v", final.Metrics)
	}
}

func TestInboundStream_ThinkingBlocks(t *testing.T) {
	body := `event: message_start
data: {"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Reasoning..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_stop
data: {"type":"message_stop"}

`
	chunks, err := collect(t, body, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	var hasThinking, hasContent bool
	for _, ch := range chunks {
		if ch.Message.Thinking == "Reasoning..." {
			hasThinking = true
		}
		if ch.Message.Content == "Answer" {
			hasContent = true
		}
	}
	if !hasThinking {
		t.Error("thinking chunk missing")
	}
	if !hasContent {
		t.Error("content chunk missing")
	}
}

func TestInboundStream_ToolUse(t *testing.T) {
	body := `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}

`
	chunks, err := collect(t, body, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	var toolCall *api.ToolCall
	for i := range chunks {
		if len(chunks[i].Message.ToolCalls) > 0 {
			toolCall = &chunks[i].Message.ToolCalls[0]
			break
		}
	}
	if toolCall == nil {
		t.Fatal("no tool call emitted")
	}
	if toolCall.ID != "toolu_1" || toolCall.Function.Name != "get_weather" {
		t.Errorf("tool call: %+v", toolCall)
	}
	city, ok := toolCall.Function.Arguments.Get("city")
	if !ok || city != "SF" {
		t.Errorf("tool args city: %v (ok=%v)", city, ok)
	}
}

func TestInboundStream_DoneSentinel(t *testing.T) {
	// OpenRouter-style trailing [DONE] marker must be accepted gracefully.
	body := `event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: message_stop
data: {"type":"message_stop"}

data: [DONE]

`
	chunks, err := collect(t, body, nil)
	if err != nil {
		t.Fatalf("Run should not error on [DONE], got: %v", err)
	}
	// Must have exactly one done chunk despite two potential termination signals.
	doneCount := 0
	for _, ch := range chunks {
		if ch.Done {
			doneCount++
		}
	}
	if doneCount != 1 {
		t.Errorf("want exactly 1 done chunk, got %d", doneCount)
	}
}

func TestInboundStream_NoMessageStopSafetyNet(t *testing.T) {
	// Upstream closes without message_stop — safety net must emit terminal done.
	body := `event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"orphan"}}

`
	chunks, err := collect(t, body, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(chunks) == 0 || !chunks[len(chunks)-1].Done {
		t.Error("safety net did not emit terminal done chunk")
	}
}

func TestInboundStream_StopReasonMapping(t *testing.T) {
	cases := []struct{ anthropic, ollama string }{
		{"end_turn", "stop"},
		{"max_tokens", "length"},
		{"tool_use", "stop"},
		{"stop_sequence", "stop"},
		{"", "stop"},
	}
	for _, tc := range cases {
		if got := mapAnthropicStopReason(tc.anthropic); got != tc.ollama {
			t.Errorf("mapAnthropicStopReason(%q) = %q, want %q", tc.anthropic, got, tc.ollama)
		}
	}
}

func TestFromMessagesResponse_CompleteConversion(t *testing.T) {
	thinkingTxt := "let me think"
	textTxt := "the answer is 42"
	resp := &MessagesResponse{
		ID:    "msg_x",
		Type:  "message",
		Role:  "assistant",
		Model: "claude-x",
		Content: []ContentBlock{
			{Type: "thinking", Thinking: &thinkingTxt},
			{Type: "text", Text: &textTxt},
			{Type: "tool_use", ID: "t_1", Name: "search", Input: api.NewToolCallFunctionArguments()},
		},
		StopReason: "tool_use",
		Usage:      Usage{InputTokens: 10, OutputTokens: 20},
	}
	chat := FromMessagesResponse(resp)
	if !chat.Done {
		t.Error("done should be true")
	}
	if chat.Message.Content != textTxt {
		t.Errorf("content: %q", chat.Message.Content)
	}
	if chat.Message.Thinking != thinkingTxt {
		t.Errorf("thinking: %q", chat.Message.Thinking)
	}
	if len(chat.Message.ToolCalls) != 1 || chat.Message.ToolCalls[0].ID != "t_1" {
		t.Errorf("tool calls: %+v", chat.Message.ToolCalls)
	}
	if chat.Metrics.PromptEvalCount != 10 || chat.Metrics.EvalCount != 20 {
		t.Errorf("metrics: %+v", chat.Metrics)
	}
}

func TestThinkParserName_Mappings(t *testing.T) {
	cases := []struct {
		id   string
		want string
	}{
		{"deepseek/deepseek-r1", "deepseek3"},
		{"deepseek/deepseek-r1-0528", "deepseek3"},
		{"tngtech/deepseek-r1t2-chimera", "deepseek3"},
		{"qwen/qwen3-vl-8b-thinking", "qwen3-vl-thinking"},
		{"qwen/qwen3-max-thinking", "qwen3-thinking"},
		{"qwen/qwen3-235b-a22b-thinking-2507", "qwen3-thinking"},
		{"allenai/olmo-3-32b-think", "olmo3-think"},
		{"moonshotai/kimi-k2-thinking", "deepseek3"},
		{"anthropic/claude-3.5-haiku", ""},
		{"openai/gpt-4o", ""},
		{"", ""},
	}
	for _, tc := range cases {
		if got := ThinkParserName(tc.id); got != tc.want {
			t.Errorf("ThinkParserName(%q) = %q, want %q", tc.id, got, tc.want)
		}
	}
}
