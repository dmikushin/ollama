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

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openrouter"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// dispatchOpenRouterChat handles an api.ChatRequest whose model resolved to
// modelSourceOpenRouter. It translates the request into OpenAI ChatCompletion
// format, POSTs it to OpenRouter, and streams the response back to the gin
// context as native Ollama ndjson chunks.
func dispatchOpenRouterChat(c *gin.Context, req *api.ChatRequest, baseModel string) {
	origModel := req.Model
	req.Model = baseModel

	client, err := openrouter.ClientFromEnv()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}

	chatReq := openrouter.ConvertChatToOllamaRequest(req)
	// Force the upstream model name to the base (stripped of :openrouter)
	chatReq.Model = baseModel

	streaming := req.Stream == nil || *req.Stream
	chatReq.Stream = streaming

	// When stream_options.include_usage is set, OpenRouter includes usage in the final chunk
	if streaming {
		chatReq.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	}

	resp, err := client.Chat(c.Request.Context(), chatReq)
	if err != nil {
		writeOpenRouterError(c, err)
		return
	}
	defer resp.Body.Close()

	contentType := "application/x-ndjson"
	if !streaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	if !streaming {
		dispatchOpenRouterNonStreaming(c, resp.Body, origModel)
		return
	}

	dispatchOpenRouterStreaming(c, resp.Body, origModel, baseModel)
}

// dispatchOpenRouterStreaming reads an OpenAI SSE stream and emits Ollama
// ndjson chunks.
func dispatchOpenRouterStreaming(c *gin.Context, body io.Reader, origModel, baseModel string) {
	_ = baseModel // reserved for parser hookup if needed later

	conv := openrouter.NewStreamConverter(origModel, func(chunk api.ChatResponse) error {
		chunk.Model = origModel
		chunk.RemoteModel = origModel
		chunk.RemoteHost = "openrouter"
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

	if err := conv.Run(body); err != nil {
		slog.Error("openrouter stream converter failed", "error", err)
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

// dispatchOpenRouterNonStreaming reads a single JSON response and converts
// it to an api.ChatResponse.
func dispatchOpenRouterNonStreaming(c *gin.Context, body io.Reader, origModel string) {
	comp, err := openrouter.ReadAndDecode(body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("openrouter: %v", err)})
		return
	}
	chat := openrouter.NonStreamingResponse(comp)
	chat.Model = origModel
	chat.RemoteModel = origModel
	chat.RemoteHost = "openrouter"
	c.JSON(http.StatusOK, chat)
}

// dispatchOpenRouterGenerate handles an api.GenerateRequest whose model
// resolved to modelSourceOpenRouter. It wraps the prompt into a single-turn
// chat, dispatches through the OpenAI-compatible translator, and reshapes
// each ChatResponse chunk back into a GenerateResponse.
func dispatchOpenRouterGenerate(c *gin.Context, req *api.GenerateRequest, baseModel string) {
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

	client, err := openrouter.ClientFromEnv()
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
		return
	}

	openaiReq := openrouter.ConvertChatToOllamaRequest(chatReq)
	openaiReq.Model = baseModel

	streaming := req.Stream == nil || *req.Stream
	openaiReq.Stream = streaming
	if streaming {
		openaiReq.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	}

	resp, err := client.Chat(c.Request.Context(), openaiReq)
	if err != nil {
		writeOpenRouterError(c, err)
		return
	}
	defer resp.Body.Close()

	contentType := "application/x-ndjson"
	if !streaming {
		contentType = "application/json; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	if !streaming {
		comp, err := openrouter.ReadAndDecode(resp.Body)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": fmt.Sprintf("openrouter: %v", err)})
			return
		}
		chat := openrouter.NonStreamingResponse(comp)
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
		gen.RemoteHost = "openrouter"
		c.JSON(http.StatusOK, gen)
		return
	}

	conv := openrouter.NewStreamConverter(origModel, func(chunk api.ChatResponse) error {
		gen := api.GenerateResponse{
			Model:       origModel,
			RemoteModel: origModel,
			RemoteHost:  "openrouter",
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

	if err := conv.Run(resp.Body); err != nil {
		slog.Error("openrouter generate stream converter failed", "error", err)
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

func writeOpenRouterError(c *gin.Context, err error) {
	var apiErr api.StatusError
	if errors.As(err, &apiErr) {
		c.JSON(apiErr.StatusCode, apiErr)
		return
	}
	c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
}

// openRouterShowResponse synthesizes an api.ShowResponse for an OpenRouter
// model. OpenRouter models are pure remote — there is no local GGUF blob.
func openRouterShowResponse(ctx context.Context, baseModel string) api.ShowResponse {
	_ = ctx // reserved for future catalog lookup

	caps := []model.Capability{model.CapabilityCompletion, model.CapabilityTools}

	modelInfo := map[string]any{
		"general.architecture": "openrouter",
		"general.basename":     baseModel,
	}

	return api.ShowResponse{
		ModelInfo: modelInfo,
		Details: api.ModelDetails{
			Format:        "remote",
			Family:        "openrouter",
			Families:      []string{"openrouter"},
			ParameterSize: "remote",
		},
		ModifiedAt:   time.Now().UTC(),
		Capabilities: caps,
		RemoteModel:  baseModel,
		RemoteHost:   "openrouter",
	}
}

// stubOpenRouterPull synthesizes an immediate-success progress stream for an
// OpenRouter model pull. There is nothing to download.
func stubOpenRouterPull(c *gin.Context, originalRef string) {
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
		slog.Warn("openrouter pull stub write failed", "model", originalRef, "error", err)
	}
	c.Writer.Flush()
}
