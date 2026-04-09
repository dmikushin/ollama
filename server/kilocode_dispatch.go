package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/types/model"
)

// dispatchKilocodeChat handles an api.ChatRequest whose model resolved to
// modelSourceKilocode. It translates the request into Anthropic Messages
// format, posts it to the KiloCode gateway, and streams the response back to
// the gin context as native Ollama ndjson chunks (or a single JSON object
// when streaming is disabled).
//
// This is the symmetric counterpart of the cloud (ollama.com) byte-proxy
// branch in ChatHandler: instead of proxying raw bytes, we translate
// protocols in both directions because KiloCode does not speak Ollama's
// native /api/chat schema.
func dispatchKilocodeChat(c *gin.Context, req *api.ChatRequest, baseModel string) {
	origModel := req.Model
	req.Model = baseModel

	client, err := anthropic.KiloCodeClientFromEnv()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}

	msgReq, err := anthropic.ToMessagesRequest(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("kilocode: translate request: %v", err)})
		return
	}

	// Force the upstream model name to match what the user typed (minus the
	// :kilocode suffix). KiloCode supports model strings like "kilo-auto/free"
	// or "anthropic/claude-3.5-sonnet" — these are passed through unchanged.
	msgReq.Model = baseModel

	streaming := req.Stream == nil || *req.Stream
	msgReq.Stream = streaming

	resp, err := client.Messages(c.Request.Context(), msgReq)
	if err != nil {
		writeKilocodeError(c, err)
		return
	}
	defer resp.Body.Close()

	contentType := "application/x-ndjson"
	if !streaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	if !streaming {
		dispatchKilocodeNonStreaming(c, resp.Body, origModel)
		return
	}

	dispatchKilocodeStreaming(c, resp.Body, origModel, baseModel)
}

// kilocodeTextParser looks up a <think>-aware parser for the given model id
// and initializes it. Returns nil for models whose text stream does not need
// tag splitting (e.g. Anthropic Claude variants that deliver structured
// thinking blocks via SSE).
func kilocodeTextParser(modelID string) parsers.Parser {
	name := anthropic.ThinkParserName(modelID)
	if name == "" {
		return nil
	}
	p := parsers.ParserForName(name)
	if p == nil {
		return nil
	}
	// The parser expects Init to be called once before Add; for the outbound
	// path we have no tools/lastMessage/thinkValue to pass, so Init receives
	// zero values purely to prime internal state.
	_ = p.Init(nil, nil, nil)
	return p
}

func dispatchKilocodeStreaming(c *gin.Context, body io.Reader, origModel, baseModel string) {
	conv := anthropic.NewInboundStreamConverter(origModel, func(chunk api.ChatResponse) error {
		// Restore the user-facing model name (with :kilocode suffix preserved)
		// so clients see the same string they asked for.
		chunk.Model = origModel
		chunk.RemoteModel = origModel
		chunk.RemoteHost = "kilocode"
		data, err := json.Marshal(chunk)
		if err != nil {
			return err
		}
		if _, err := c.Writer.Write(append(data, '\n')); err != nil {
			return err
		}
		c.Writer.Flush()
		return nil
	})
	conv.TextParser = kilocodeTextParser(baseModel)

	if err := conv.Run(body); err != nil {
		slog.Error("kilocode stream converter failed", "error", err)
		// Best-effort: emit a final done chunk so clients aren't left hanging.
		final := api.ChatResponse{
			Model:      origModel,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "stop",
		}
		if data, mErr := json.Marshal(final); mErr == nil {
			_, _ = c.Writer.Write(append(data, '\n'))
			c.Writer.Flush()
		}
	}
}

func dispatchKilocodeNonStreaming(c *gin.Context, body io.Reader, origModel string) {
	raw, err := io.ReadAll(io.LimitReader(body, 16<<20))
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("kilocode: read response: %v", err)})
		return
	}
	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("kilocode: decode response: %v", err)})
		return
	}
	chat := anthropic.FromMessagesResponse(&resp)
	chat.Model = origModel
	chat.RemoteModel = origModel
	chat.RemoteHost = "kilocode"
	c.JSON(http.StatusOK, chat)
}

// dispatchKilocodeGenerate handles an api.GenerateRequest whose model
// resolved to modelSourceKilocode. It wraps the prompt into a single-turn
// chat, dispatches through the same Anthropic translator, and reshapes each
// ChatResponse chunk back into a GenerateResponse so the native ollama CLI
// (`ollama run <model> <prompt>`) sees a familiar streaming format.
func dispatchKilocodeGenerate(c *gin.Context, req *api.GenerateRequest, baseModel string) {
	origModel := req.Model

	var messages []api.Message
	if req.System != "" {
		messages = append(messages, api.Message{Role: "system", Content: req.System})
	}
	if req.Prompt != "" {
		messages = append(messages, api.Message{Role: "user", Content: req.Prompt})
	}
	chatReq := &api.ChatRequest{
		Model:    baseModel,
		Messages: messages,
		Options:  req.Options,
		Stream:   req.Stream,
		Think:    req.Think,
	}

	client, err := anthropic.KiloCodeClientFromEnv()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}

	msgReq, err := anthropic.ToMessagesRequest(chatReq)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("kilocode: translate request: %v", err)})
		return
	}
	msgReq.Model = baseModel

	streaming := req.Stream == nil || *req.Stream
	msgReq.Stream = streaming

	resp, err := client.Messages(c.Request.Context(), msgReq)
	if err != nil {
		writeKilocodeError(c, err)
		return
	}
	defer resp.Body.Close()

	contentType := "application/x-ndjson"
	if !streaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	if !streaming {
		raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("kilocode: read response: %v", err)})
			return
		}
		var ar anthropic.MessagesResponse
		if err := json.Unmarshal(raw, &ar); err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("kilocode: decode response: %v", err)})
			return
		}
		chat := anthropic.FromMessagesResponse(&ar)
		gen := api.GenerateResponse{
			Model:      origModel,
			CreatedAt:  chat.CreatedAt,
			Response:   chat.Message.Content,
			Thinking:   chat.Message.Thinking,
			Done:       true,
			DoneReason: chat.DoneReason,
			Metrics:    chat.Metrics,
		}
		gen.RemoteModel = origModel
		gen.RemoteHost = "kilocode"
		c.JSON(http.StatusOK, gen)
		return
	}

	conv := anthropic.NewInboundStreamConverter(origModel, func(chunk api.ChatResponse) error {
		gen := api.GenerateResponse{
			Model:       origModel,
			RemoteModel: origModel,
			RemoteHost:  "kilocode",
			CreatedAt:   chunk.CreatedAt,
			Response:    chunk.Message.Content,
			Thinking:    chunk.Message.Thinking,
			Done:        chunk.Done,
			DoneReason:  chunk.DoneReason,
			Metrics:     chunk.Metrics,
		}
		data, err := json.Marshal(gen)
		if err != nil {
			return err
		}
		if _, err := c.Writer.Write(append(data, '\n')); err != nil {
			return err
		}
		c.Writer.Flush()
		return nil
	})
	conv.TextParser = kilocodeTextParser(baseModel)

	if err := conv.Run(resp.Body); err != nil {
		slog.Error("kilocode generate stream converter failed", "error", err)
		final := api.GenerateResponse{
			Model:      origModel,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "stop",
		}
		if data, mErr := json.Marshal(final); mErr == nil {
			_, _ = c.Writer.Write(append(data, '\n'))
			c.Writer.Flush()
		}
	}
}

func writeKilocodeError(c *gin.Context, err error) {
	var apiErr api.StatusError
	if errors.As(err, &apiErr) {
		c.JSON(apiErr.StatusCode, apiErr)
		return
	}
	c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
}

// kilocodeShowResponse synthesizes an api.ShowResponse for a KiloCode model
// using metadata from the upstream catalog when available. KiloCode models
// live entirely upstream — there is no local manifest or GGUF blob — so the
// show endpoint just echoes back enough metadata to satisfy callers like the
// `ollama run` CLI which checks for existence before issuing a chat request.
//
// Capabilities are inferred from the catalog entry's supported_parameters:
//   - tools         → CapabilityTools
//   - reasoning     → CapabilityThinking
//   - image input   → CapabilityVision
//
// If the catalog fetch fails (network error, invalid key, etc.) we fall back
// to the safe subset (completion + tools) so the CLI can still proceed.
func kilocodeShowResponse(ctx context.Context, baseModel string) api.ShowResponse {
	caps := []model.Capability{model.CapabilityCompletion, model.CapabilityTools}
	contextLen := 0
	var description string
	modelInfo := map[string]any{
		"general.architecture": "kilocode",
		"general.basename":     baseModel,
	}

	if info, err := anthropic.DefaultCatalog.Lookup(ctx, baseModel); err == nil && info != nil {
		description = info.Description
		_ = description // surfaced via ModelInfo below
		contextLen = info.ContextLength
		// Tools is enabled by default above; only toggle off if the catalog
		// says the model does not support them.
		if !info.HasTool() {
			caps = []model.Capability{model.CapabilityCompletion}
		}
		if info.HasReasoning() {
			caps = append(caps, model.CapabilityThinking)
		}
		if info.HasVision() {
			caps = append(caps, model.CapabilityVision)
		}
		if contextLen > 0 {
			modelInfo["kilocode.context_length"] = contextLen
		}
		if info.MaxCompletionTokens > 0 {
			modelInfo["kilocode.max_completion_tokens"] = info.MaxCompletionTokens
		}
		if description != "" {
			modelInfo["kilocode.description"] = description
		}
	}

	return api.ShowResponse{
		ModelInfo: modelInfo,
		Details: api.ModelDetails{
			Format:        "remote",
			Family:        "kilocode",
			Families:      []string{"kilocode"},
			ParameterSize: "remote",
		},
		ModifiedAt:   time.Now().UTC(),
		Capabilities: caps,
		RemoteModel:  baseModel,
		RemoteHost:   "kilocode",
	}
}

// stubKilocodePull synthesizes an immediate-success progress stream for a
// KiloCode model pull. There is nothing to download; we emit a single
// "success" status so the CLI's progress bar finishes cleanly.
func stubKilocodePull(c *gin.Context, originalRef string) {
	c.Header("Content-Type", "application/x-ndjson")
	progress := api.ProgressResponse{
		Status: "success",
	}
	data, err := json.Marshal(progress)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if _, err := c.Writer.Write(append(data, '\n')); err != nil {
		slog.Warn("kilocode pull stub write failed", "model", originalRef, "error", err)
	}
	c.Writer.Flush()
}
