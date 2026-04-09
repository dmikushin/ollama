package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

// InboundStreamConverter parses an Anthropic Messages streaming SSE response
// and emits incremental api.ChatResponse chunks via the provided callback. It
// is the inverse of StreamConverter: SSE events → Ollama native streaming.
//
// The caller is responsible for providing a Reader connected to the HTTP
// response body (usually *http.Response.Body). The converter closes neither
// the reader nor the response.
type InboundStreamConverter struct {
	Model      string
	CreatedAt  time.Time
	OnResponse func(api.ChatResponse) error

	// TextParser, when non-nil, is fed every text_delta before the content is
	// emitted downstream. It splits raw text into visible content, hidden
	// thinking, and tool calls — used for reasoning models whose upstream
	// path delivers raw <think>...</think> tags rather than structured
	// Anthropic thinking blocks. Leave nil for passthrough behavior.
	TextParser parsers.Parser

	// Per-block state accumulators, keyed by Anthropic content block index.
	toolCalls map[int]*toolCallAccumulator
	// Token usage reported by the upstream on message_start / message_delta.
	inputTokens  int
	outputTokens int
	// stopReason is captured from message_delta and used in the final done
	// ChatResponse.
	stopReason string
	// finalEmitted guards against double-emission of the terminal done chunk
	// if the upstream sends both message_stop and a trailing dispatch.
	finalEmitted bool
}

type toolCallAccumulator struct {
	id       string
	name     string
	jsonBuf  strings.Builder
	finished bool
}

// NewInboundStreamConverter constructs an InboundStreamConverter. Callers
// must set OnResponse before invoking Run.
func NewInboundStreamConverter(model string, onResponse func(api.ChatResponse) error) *InboundStreamConverter {
	return &InboundStreamConverter{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		OnResponse: onResponse,
		toolCalls:  make(map[int]*toolCallAccumulator),
	}
}

// Run reads the SSE stream from r until EOF or context cancellation and
// drives OnResponse with incremental ChatResponse chunks, followed by a final
// chunk with Done=true. Returns nil on graceful completion.
func (c *InboundStreamConverter) Run(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	// Allow large SSE frames (tool call JSON can be many KB).
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 4*1024*1024)

	var eventName string
	var dataLines []string

	dispatch := func() error {
		if eventName == "" && len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		eventName = ""
		dataLines = nil
		if data == "" {
			return nil
		}
		return c.handleEvent(data)
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			// Blank line terminates an event per SSE spec.
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			// Comment / keepalive.
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			continue
		}
		// Unknown field; ignore per SSE spec.
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read sse stream: %w", err)
	}
	// Flush any trailing event that didn't end with a blank line.
	if err := dispatch(); err != nil {
		return err
	}
	// Safety net: if upstream ended the stream without a message_stop event,
	// synthesize the terminal done chunk so clients don't hang waiting.
	return c.emitFinal()
}

// handleEvent decodes a single SSE data payload and dispatches based on the
// "type" field. Anthropic prefixes each payload with a type discriminator, so
// we key off that rather than the SSE event name (the two are in sync but
// we only need one source of truth).
func (c *InboundStreamConverter) handleEvent(data string) error {
	// OpenAI-compatible gateways (KiloCode, OpenRouter) append a sentinel
	// "data: [DONE]" frame after the final message_stop. It is not valid
	// JSON; treat it as end-of-stream.
	if data == "[DONE]" {
		return c.emitFinal()
	}
	var peek struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(data), &peek); err != nil {
		return fmt.Errorf("decode sse event: %w", err)
	}

	switch peek.Type {
	case "message_start":
		var ev MessageStartEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		c.inputTokens = ev.Message.Usage.InputTokens
		c.outputTokens = ev.Message.Usage.OutputTokens
		if ev.Message.Model != "" {
			c.Model = ev.Message.Model
		}
		return nil

	case "content_block_start":
		var ev ContentBlockStartEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		if ev.ContentBlock.Type == "tool_use" {
			c.toolCalls[ev.Index] = &toolCallAccumulator{
				id:   ev.ContentBlock.ID,
				name: ev.ContentBlock.Name,
			}
		}
		return nil

	case "content_block_delta":
		var ev ContentBlockDeltaEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		return c.emitDelta(ev)

	case "content_block_stop":
		var ev ContentBlockStopEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		return c.finalizeBlock(ev.Index)

	case "message_delta":
		var ev MessageDeltaEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		if ev.Delta.StopReason != "" {
			c.stopReason = ev.Delta.StopReason
		}
		if ev.Usage.OutputTokens > 0 {
			c.outputTokens = ev.Usage.OutputTokens
		}
		if ev.Usage.InputTokens > 0 {
			c.inputTokens = ev.Usage.InputTokens
		}
		return nil

	case "message_stop":
		return c.emitFinal()

	case "ping":
		return nil

	case "error":
		var ev StreamErrorEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		return fmt.Errorf("upstream stream error: %s: %s", ev.Error.Type, ev.Error.Message)
	}

	return nil
}

// emitDelta turns a content_block_delta into an incremental ChatResponse.
// Text and thinking deltas are emitted immediately; tool call JSON deltas are
// accumulated and emitted as a single complete tool_call in finalizeBlock.
func (c *InboundStreamConverter) emitDelta(ev ContentBlockDeltaEvent) error {
	switch ev.Delta.Type {
	case "text_delta":
		if ev.Delta.Text == "" {
			return nil
		}
		return c.emitTextThroughParser(ev.Delta.Text, false)

	case "thinking_delta":
		if ev.Delta.Thinking == "" {
			return nil
		}
		return c.OnResponse(api.ChatResponse{
			Model:     c.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:     "assistant",
				Thinking: ev.Delta.Thinking,
			},
		})

	case "input_json_delta":
		acc, ok := c.toolCalls[ev.Index]
		if !ok {
			// Delta arrived without a matching content_block_start; ignore.
			return nil
		}
		acc.jsonBuf.WriteString(ev.Delta.PartialJSON)
		return nil

	case "signature_delta":
		// Anthropic emits a signature on thinking blocks for encrypted
		// reasoning traces. Ollama's native schema has nowhere to carry it,
		// so we drop it. The visible thinking text has already been emitted.
		return nil
	}
	return nil
}

// finalizeBlock flushes the accumulator for a content block when its
// content_block_stop event arrives. For tool_use blocks this is where we
// build the complete api.ToolCall and emit it to the caller.
func (c *InboundStreamConverter) finalizeBlock(index int) error {
	acc, ok := c.toolCalls[index]
	if !ok {
		return nil
	}
	if acc.finished {
		return nil
	}
	acc.finished = true

	args := api.NewToolCallFunctionArguments()
	raw := strings.TrimSpace(acc.jsonBuf.String())
	if raw != "" {
		if err := json.Unmarshal([]byte(raw), &args); err != nil {
			return fmt.Errorf("decode tool call arguments for %q: %w", acc.name, err)
		}
	}

	return c.OnResponse(api.ChatResponse{
		Model:     c.Model,
		CreatedAt: time.Now().UTC(),
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: acc.id,
					Function: api.ToolCallFunction{
						Name:      acc.name,
						Arguments: args,
					},
				},
			},
		},
	})
}

// emitTextThroughParser pipes a text delta through the optional TextParser
// before emitting a ChatResponse. When no parser is configured it degrades
// to a simple content-delta passthrough. When a parser is configured, the
// parser may return content, extracted thinking, and/or tool calls — each
// non-empty channel is flushed as its own ChatResponse chunk so downstream
// ndjson consumers see the same shape as local runner output.
func (c *InboundStreamConverter) emitTextThroughParser(text string, done bool) error {
	if c.TextParser == nil {
		if text == "" {
			return nil
		}
		return c.OnResponse(api.ChatResponse{
			Model:     c.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:    "assistant",
				Content: text,
			},
		})
	}

	content, thinking, calls, err := c.TextParser.Add(text, done)
	if err != nil {
		return fmt.Errorf("text parser: %w", err)
	}
	if thinking != "" {
		if err := c.OnResponse(api.ChatResponse{
			Model:     c.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:     "assistant",
				Thinking: thinking,
			},
		}); err != nil {
			return err
		}
	}
	if content != "" {
		if err := c.OnResponse(api.ChatResponse{
			Model:     c.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:    "assistant",
				Content: content,
			},
		}); err != nil {
			return err
		}
	}
	if len(calls) > 0 {
		if err := c.OnResponse(api.ChatResponse{
			Model:     c.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:      "assistant",
				ToolCalls: calls,
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

// emitFinal produces the terminal ChatResponse chunk (Done=true) with the
// mapped DoneReason and accumulated token usage metrics. It is idempotent:
// subsequent calls are no-ops so duplicate message_stop events do not cause
// duplicate done chunks downstream.
func (c *InboundStreamConverter) emitFinal() error {
	if c.finalEmitted {
		return nil
	}
	c.finalEmitted = true
	// Drain any content buffered inside the TextParser (e.g. trailing
	// partial <think> block) before emitting the terminal done chunk.
	if c.TextParser != nil {
		if err := c.emitTextThroughParser("", true); err != nil {
			return err
		}
	}
	doneReason := mapAnthropicStopReason(c.stopReason)
	return c.OnResponse(api.ChatResponse{
		Model:      c.Model,
		CreatedAt:  time.Now().UTC(),
		Message:    api.Message{Role: "assistant"},
		Done:       true,
		DoneReason: doneReason,
		Metrics: api.Metrics{
			PromptEvalCount: c.inputTokens,
			EvalCount:       c.outputTokens,
		},
	})
}

// mapAnthropicStopReason converts an Anthropic stop_reason into Ollama's
// DoneReason nomenclature. Inverse of mapStopReason.
func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "stop"
	case "stop_sequence":
		return "stop"
	case "":
		return "stop"
	}
	return reason
}

// FromMessagesResponse converts a complete (non-streaming) Anthropic
// MessagesResponse into a single api.ChatResponse with Done=true. Used when
// the client requested stream=false and we received the full JSON body.
func FromMessagesResponse(resp *MessagesResponse) api.ChatResponse {
	msg := api.Message{Role: "assistant"}
	var textParts []string
	var thinkingParts []string

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != nil {
				textParts = append(textParts, *block.Text)
			}
		case "thinking":
			if block.Thinking != nil {
				thinkingParts = append(thinkingParts, *block.Thinking)
			}
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
				ID: block.ID,
				Function: api.ToolCallFunction{
					Name:      block.Name,
					Arguments: block.Input,
				},
			})
		}
	}

	msg.Content = strings.Join(textParts, "")
	msg.Thinking = strings.Join(thinkingParts, "")

	return api.ChatResponse{
		Model:      resp.Model,
		CreatedAt:  time.Now().UTC(),
		Message:    msg,
		Done:       true,
		DoneReason: mapAnthropicStopReason(resp.StopReason),
		Metrics: api.Metrics{
			PromptEvalCount: resp.Usage.InputTokens,
			EvalCount:       resp.Usage.OutputTokens,
		},
	}
}
