package openrouter

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/openai"
)

// StreamConverter reads an OpenAI-compatible SSE stream from OpenRouter
// and converts it to Ollama's api.ChatResponse chunks.
type StreamConverter struct {
	Model    string
	OnChunk  func(chunk api.ChatResponse) error
	// TextParser, when non-nil, feeds content through a model-specific
	// parser (e.g. for <think> tag splitting). Used for reasoning models
	// that emit raw tags rather than structured delta.Reasoning.
	TextParser parsers.Parser

	// State tracking for usage and final done
	doneReason       string
	promptTokens     int
	completionTokens int
	totalTokens      int
	hasEmittedDone   bool
	// Set to true once we see structured delta.Reasoning from the API.
	// When true, TextParser is bypassed for content because OpenRouter
	// already split thinking from content (no raw <think> tags).
	hasStructuredReasoning bool
	// Set to true once real content has been emitted during streaming.
	hasEmittedContent bool
	// Accumulated thinking text, used to populate content in the done
	// chunk for models that only emit reasoning and never send delta.Content.
	thinkingAccum strings.Builder
	// Pending tool calls accumulated from incremental deltas.
	// OpenAI SSE sends tool calls as: first delta has ID+name,
	// subsequent deltas append to arguments.  Keyed by delta index.
	pendingToolCalls map[int]*pendingToolCall
}

type pendingToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

// NewStreamConverter creates a converter that will call onChunk for each
// converted chunk.
func NewStreamConverter(model string, onChunk func(chunk api.ChatResponse) error) *StreamConverter {
	return &StreamConverter{
		Model:   model,
		OnChunk: onChunk,
	}
}

// Run reads the SSE stream and converts each event to an api.ChatResponse chunk.
func (sc *StreamConverter) Run(reader io.Reader) error {
	scanner := bufio.NewScanner(reader)
	// Increase buffer size for large SSE events
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)

	var eventBuf strings.Builder
	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			// Empty line signals end of event — process accumulated data
			if eventBuf.Len() > 0 {
				if err := sc.processEvent(eventBuf.String()); err != nil {
					return err
				}
				eventBuf.Reset()
			}
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				// Final sentinel from OpenAI-compatible streams
				if err := sc.emitDone(); err != nil {
					return err
				}
				return nil
			}
			eventBuf.WriteString(data)
			eventBuf.WriteByte('\n')
		}
		// Ignore "event:" and "id:" lines — we only care about data payloads
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("reading SSE stream: %w", err)
	}

	// Process any remaining buffered event
	if eventBuf.Len() > 0 {
		if err := sc.processEvent(eventBuf.String()); err != nil {
			return err
		}
	}

	// Safety net: if the stream closed without a [DONE] sentinel, emit a final done chunk
	if !sc.hasEmittedDone {
		if err := sc.emitDone(); err != nil {
			return err
		}
	}

	return nil
}

// contentToString extracts a string from the Message.Content field (which is any).
func contentToString(c any) string {
	if c == nil {
		return ""
	}
	if s, ok := c.(string); ok {
		return s
	}
	// If it's a slice (multi-modal content), extract text portions
	if slice, ok := c.([]any); ok {
		var parts []string
		for _, item := range slice {
			if s, ok := item.(string); ok {
				parts = append(parts, s)
			} else if m, ok := item.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "")
	}
	return fmt.Sprintf("%v", c)
}

func (sc *StreamConverter) processEvent(dataStr string) error {
	// Trim trailing newlines that we accumulated
	dataStr = strings.TrimSpace(dataStr)
	if dataStr == "" {
		return nil
	}

	var chunk openai.ChatCompletionChunk
	if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
		return fmt.Errorf("parse SSE chunk: %w", err)
	}

	// OpenRouter streams may include usage at the end when stream_options is set
	if chunk.Usage != nil {
		sc.promptTokens = chunk.Usage.PromptTokens
		sc.completionTokens = chunk.Usage.CompletionTokens
		sc.totalTokens = chunk.Usage.TotalTokens
		return nil
	}

	// Skip chunks with no choices
	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Handle thinking/reasoning via structured delta first (priority).
	// OpenRouter models like qwen3.6-plus put the ENTIRE answer in
	// delta.Reasoning with no delta.Content.  Emit as Content so
	// downstream Anthropic SSE produces text_delta (visible response)
	// instead of thinking_delta (hidden reasoning panel).
	if delta.Reasoning != "" {
		sc.hasStructuredReasoning = true
		sc.thinkingAccum.WriteString(delta.Reasoning)
		sc.hasEmittedContent = true
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
			CreatedAt: time.Unix(chunk.Created, 0).UTC(),
			Message: api.Message{
				Role:    "assistant",
				Content: delta.Reasoning,
			},
		}); err != nil {
			return err
		}
	}

	contentStr := contentToString(delta.Content)

	// If a TextParser is configured AND the stream does NOT use structured
	// reasoning, route content through the parser for <think> tag splitting.
	// When OpenRouter already provides structured delta.Reasoning, the
	// content arrives clean — bypass the parser which would buffer forever
	// waiting for </think> tags that will never come.
	if sc.TextParser != nil && contentStr != "" && !sc.hasStructuredReasoning {
		if err := sc.emitThroughParser(contentStr, false); err != nil {
			return err
		}
		sc.hasEmittedContent = true
	} else if contentStr != "" {
		sc.hasEmittedContent = true
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
			CreatedAt: time.Unix(chunk.Created, 0).UTC(),
			Message: api.Message{
				Role:    "assistant",
				Content: contentStr,
			},
		}); err != nil {
			return err
		}
	}

	// Accumulate tool call deltas (OpenAI streams them incrementally:
	// first chunk has ID+name, subsequent chunks append arguments).
	if len(delta.ToolCalls) > 0 {
		if sc.pendingToolCalls == nil {
			sc.pendingToolCalls = make(map[int]*pendingToolCall)
		}
		for _, tc := range delta.ToolCalls {
			idx := tc.Index
			ptc, ok := sc.pendingToolCalls[idx]
			if !ok {
				ptc = &pendingToolCall{}
				sc.pendingToolCalls[idx] = ptc
			}
			if tc.ID != "" {
				ptc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				ptc.Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				ptc.Args.WriteString(tc.Function.Arguments)
			}
		}
	}

	// Handle finish reason
	if choice.FinishReason != nil {
		sc.doneReason = *choice.FinishReason
		// Usage metrics (from stream_options) arrive in a separate usage-only
		// chunk AFTER the finish_reason chunk.  Defer emitting the done
		// chunk to the next Run iteration or the [DONE] sentinel so that
		// metrics are captured.
		return nil
	}

	return nil
}

func (sc *StreamConverter) emitDone() error {
	if sc.hasEmittedDone {
		return nil
	}
	sc.hasEmittedDone = true
	// Drain any pending content from the parser before emitting done.
	if sc.TextParser != nil {
		if err := sc.emitThroughParser("", true); err != nil {
			return err
		}
	}
	// Emit accumulated tool calls as a single chunk with complete arguments.
	if len(sc.pendingToolCalls) > 0 {
		tcs := make([]api.ToolCall, 0, len(sc.pendingToolCalls))
		for _, ptc := range sc.pendingToolCalls {
			tc := api.ToolCall{
				ID: ptc.ID,
				Function: api.ToolCallFunction{
					Name: ptc.Name,
				},
			}
			if args := ptc.Args.String(); args != "" {
				json.Unmarshal([]byte(args), &tc.Function.Arguments)
			}
			tcs = append(tcs, tc)
		}
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:      "assistant",
				ToolCalls: tcs,
			},
		}); err != nil {
			return err
		}
	}
	resp := api.ChatResponse{
		Model:      sc.Model,
		CreatedAt:  time.Now().UTC(),
		Message:    api.Message{Role: "assistant"},
		Done:       true,
		DoneReason: sc.doneReason,
	}
	if resp.DoneReason == "" {
		resp.DoneReason = "stop"
	}
	// If the model never emitted visible content (only reasoning/thinking),
	// use the accumulated thinking as content so downstream consumers see
	// the answer.
	if !sc.hasEmittedContent {
		if allThinking := sc.thinkingAccum.String(); allThinking != "" {
			resp.Message.Content = allThinking
		}
	}
	resp.Metrics.PromptEvalCount = sc.promptTokens
	resp.Metrics.EvalCount = sc.completionTokens
	return sc.OnChunk(resp)
}

// emitThroughParser pipes content through the TextParser before emitting.
// When no parser is configured it emits content verbatim. When a parser is
// configured, it splits into thinking, content, and tool calls.
func (sc *StreamConverter) emitThroughParser(text string, done bool) error {
	if sc.TextParser == nil {
		if text == "" {
			return nil
		}
		return sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
			CreatedAt: time.Now().UTC(),
			Message: api.Message{
				Role:    "assistant",
				Content: text,
			},
		})
	}

	content, thinking, calls, err := sc.TextParser.Add(text, done)
	if err != nil {
		return fmt.Errorf("text parser: %w", err)
	}
	if thinking != "" {
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
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
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
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
		if err := sc.OnChunk(api.ChatResponse{
			Model:     sc.Model,
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

// NonStreamingResponse converts a non-streaming ChatCompletion to a single
// api.ChatResponse with Done=true.
func NonStreamingResponse(comp *openai.ChatCompletion) api.ChatResponse {
	resp := api.ChatResponse{
		Model:     comp.Model,
		CreatedAt: time.Unix(comp.Created, 0).UTC(),
		Done:      true,
	}

	if len(comp.Choices) > 0 {
		msg := comp.Choices[0].Message
		resp.Message.Role = msg.Role
		resp.Message.Content = contentToString(msg.Content)
		resp.Message.Thinking = msg.Reasoning
		resp.Message.ToolCalls = toOllamaToolCalls(msg.ToolCalls)

		if comp.Choices[0].FinishReason != nil {
			resp.DoneReason = *comp.Choices[0].FinishReason
		}
	}

	resp.Metrics.PromptEvalCount = comp.Usage.PromptTokens
	resp.Metrics.EvalCount = comp.Usage.CompletionTokens

	return resp
}

func toOllamaToolCalls(tc []openai.ToolCall) []api.ToolCall {
	result := make([]api.ToolCall, len(tc))
	for i, t := range tc {
		result[i].ID = t.ID
		result[i].Function.Name = t.Function.Name
		result[i].Function.Index = t.Index
		if t.Function.Arguments != "" {
			json.Unmarshal([]byte(t.Function.Arguments), &result[i].Function.Arguments)
		}
	}
	return result
}

// ReadAndDecode reads the full response body and decodes it as a ChatCompletion.
func ReadAndDecode(body io.Reader) (*openai.ChatCompletion, error) {
	data, err := io.ReadAll(io.LimitReader(body, 16*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	var comp openai.ChatCompletion
	if err := json.Unmarshal(data, &comp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &comp, nil
}

func toStringSlice(v any) []string {
	if v == nil {
		return nil
	}
	if ss, ok := v.([]string); ok {
		return ss
	}
	if anySlice, ok := v.([]any); ok {
		out := make([]string, 0, len(anySlice))
		for _, item := range anySlice {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	if s, ok := v.(string); ok {
		return []string{s}
	}
	return nil
}

// ConvertChatToOllamaRequest converts an api.ChatRequest to an OpenAI ChatCompletionRequest
func ConvertChatToOllamaRequest(req *api.ChatRequest) *openai.ChatCompletionRequest {
	messages := make([]openai.Message, len(req.Messages))
	for i, msg := range req.Messages {
		m := openai.Message{
			Role:       msg.Role,
			Content:    msg.Content,
			Reasoning:  msg.Thinking,
			ToolCallID: msg.ToolCallID,
		}
		for _, tc := range msg.ToolCalls {
			args, _ := json.Marshal(tc.Function.Arguments)
			m.ToolCalls = append(m.ToolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      tc.Function.Name,
					Arguments: string(args),
				},
			})
		}
		messages[i] = m
	}

	streaming := req.Stream == nil || *req.Stream

	ocr := &openai.ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   streaming,
		Tools:    req.Tools,
	}

	if req.Options != nil {
		if temp, ok := req.Options["temperature"].(float64); ok {
			ocr.Temperature = &temp
		}
		if topP, ok := req.Options["top_p"].(float64); ok {
			ocr.TopP = &topP
		}
		if numPredict, ok := req.Options["num_predict"].(float64); ok {
			n := int(numPredict)
			ocr.MaxTokens = &n
		}
		if stop := toStringSlice(req.Options["stop"]); len(stop) > 0 {
			// Convert to []any for the openai.Request
			stopAny := make([]any, len(stop))
			for i, s := range stop {
				stopAny[i] = s
			}
			ocr.Stop = stopAny
		}
	}

	if len(req.Format) > 0 {
		var formatVal string
		json.Unmarshal(req.Format, &formatVal)
		if strings.EqualFold(formatVal, "json") {
			ocr.ResponseFormat = &openai.ResponseFormat{Type: "json_object"}
		}
	}

	if req.Think != nil {
		switch v := req.Think.Value.(type) {
		case bool:
			if v {
				ocr.Reasoning = &openai.Reasoning{Effort: "medium"}
			}
		case string:
			ocr.Reasoning = &openai.Reasoning{Effort: v}
		}
	}

	return ocr
}

// ConvertGenerateToOllamaRequest converts an api.GenerateRequest to an OpenAI ChatCompletionRequest
func ConvertGenerateToOllamaRequest(req *api.GenerateRequest) *openai.ChatCompletionRequest {
	messages := []openai.Message{}
	if req.System != "" {
		messages = append(messages, openai.Message{Role: "system", Content: req.System})
	}
	if req.Prompt != "" {
		messages = append(messages, openai.Message{Role: "user", Content: req.Prompt})
	}

	streaming := req.Stream == nil || *req.Stream

	ocr := &openai.ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   streaming,
	}

	if req.Options != nil {
		if temp, ok := req.Options["temperature"].(float64); ok {
			ocr.Temperature = &temp
		}
		if topP, ok := req.Options["top_p"].(float64); ok {
			ocr.TopP = &topP
		}
		if numPredict, ok := req.Options["num_predict"].(float64); ok {
			n := int(numPredict)
			ocr.MaxTokens = &n
		}
	}

	return ocr
}
