package openrouter

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// StreamConverter reads an OpenAI-compatible SSE stream from OpenRouter
// and converts it to Ollama's api.ChatResponse chunks.
type StreamConverter struct {
	Model    string
	OnChunk  func(chunk api.ChatResponse) error
	// State tracking for usage and final done
	doneReason       string
	promptTokens     int
	completionTokens int
	totalTokens      int
	hasEmittedDone   bool
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
				sc.hasEmittedDone = true
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

	contentStr := contentToString(delta.Content)

	resp := api.ChatResponse{
		Model:     sc.Model,
		CreatedAt: time.Unix(chunk.Created, 0).UTC(),
		Message: api.Message{
			Role:    "assistant",
			Content: contentStr,
		},
	}

	// Handle thinking/reasoning content
	if delta.Reasoning != "" {
		resp.Message.Thinking = delta.Reasoning
	}

	// Handle tool calls
	if len(delta.ToolCalls) > 0 {
		resp.Message.ToolCalls = toOllamaToolCalls(delta.ToolCalls)
	}

	// Handle finish reason
	if choice.FinishReason != nil {
		sc.doneReason = *choice.FinishReason
		resp.Done = true
		resp.DoneReason = sc.doneReason
		resp.Metrics.PromptEvalCount = sc.promptTokens
		resp.Metrics.EvalCount = sc.completionTokens
	}

	return sc.OnChunk(resp)
}

func (sc *StreamConverter) emitDone() error {
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
	resp.Metrics.PromptEvalCount = sc.promptTokens
	resp.Metrics.EvalCount = sc.completionTokens
	return sc.OnChunk(resp)
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
		messages[i] = openai.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
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
