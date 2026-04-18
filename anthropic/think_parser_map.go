package anthropic

import "strings"

// ThinkParserName returns the name of a model/parsers parser to apply to
// plain text output from the given KiloCode/OpenRouter model id, or the
// empty string if no parser is applicable.
//
// This mapping exists because some reasoning models (DeepSeek R1, Qwen3
// thinking variants, Olmo3-think, etc.) emit raw <think>...</think> tags or
// similar discriminators in their text stream when routed through an
// OpenRouter-style gateway that does not transform reasoning into structured
// Anthropic "thinking" content blocks. Running the text through the matching
// parser peels the reasoning out of the visible content so downstream
// clients get a clean content/thinking split.
//
// Models that emit structured Anthropic thinking blocks (Anthropic's own
// Claude family, any model where the gateway already normalizes reasoning)
// must NOT be listed here — routing their output through a tag parser would
// leave Anthropic's thinking blocks in place AND double-emit any stray
// plain-text reasoning. The contract is: only map models whose TEXT path
// contains raw reasoning tags.
func ThinkParserName(modelID string) string {
	id := strings.ToLower(strings.TrimSpace(modelID))
	if id == "" {
		return ""
	}

	// DeepSeek R1 family — emits <think>...</think> before visible answer.
	if strings.HasPrefix(id, "deepseek/deepseek-r1") ||
		strings.Contains(id, "deepseek-r1") ||
		strings.Contains(id, "r1t2-chimera") {
		return "deepseek3"
	}

	// Qwen3 / Qwen3.5 thinking variants.
	// qwen3.6-plus is a reasoning model that emits <think> tags despite not
	// having "thinking" in its name — it needs the qwen3-thinking parser.
	if strings.HasPrefix(id, "qwen/qwen3-vl-") && strings.Contains(id, "thinking") {
		return "qwen3-vl-thinking"
	}
	if strings.HasPrefix(id, "qwen/") && strings.Contains(id, "thinking") {
		return "qwen3-thinking"
	}
	// Allen AI Olmo 3 think variants.
	if strings.HasPrefix(id, "allenai/olmo-3-") && strings.Contains(id, "think") {
		return "olmo3-think"
	}

	// Moonshot Kimi thinking — same <think>...</think> convention.
	if strings.Contains(id, "kimi-k2-thinking") || strings.Contains(id, "kimi-k2.5-thinking") {
		return "deepseek3"
	}

	return ""
}
