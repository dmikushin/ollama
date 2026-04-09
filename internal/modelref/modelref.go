package modelref

import (
	"errors"
	"fmt"
	"strings"
)

type ModelSource uint8

const (
	ModelSourceUnspecified ModelSource = iota
	ModelSourceLocal
	ModelSourceCloud
	// ModelSourceKilocode routes inference through the KiloCode Gateway,
	// which exposes an Anthropic-compatible /v1/messages endpoint. Ollama
	// acts as a protocol translator between its native /api/chat API and
	// the Anthropic Messages API.
	ModelSourceKilocode
)

var (
	ErrConflictingSourceSuffix = errors.New("use only one of :local, :cloud, or :kilocode")
	ErrModelRequired           = errors.New("model is required")
)

type ParsedRef struct {
	Original string
	Base     string
	Source   ModelSource
}

func ParseRef(raw string) (ParsedRef, error) {
	var zero ParsedRef

	raw = strings.TrimSpace(raw)
	if raw == "" {
		return zero, ErrModelRequired
	}

	base, source, explicit := parseSourceSuffix(raw)
	if explicit {
		if _, _, nested := parseSourceSuffix(base); nested {
			return zero, fmt.Errorf("%w: %q", ErrConflictingSourceSuffix, raw)
		}
	}

	return ParsedRef{
		Original: raw,
		Base:     base,
		Source:   source,
	}, nil
}

func HasExplicitCloudSource(raw string) bool {
	parsedRef, err := ParseRef(raw)
	return err == nil && parsedRef.Source == ModelSourceCloud
}

func HasExplicitLocalSource(raw string) bool {
	parsedRef, err := ParseRef(raw)
	return err == nil && parsedRef.Source == ModelSourceLocal
}

func StripCloudSourceTag(raw string) (string, bool) {
	parsedRef, err := ParseRef(raw)
	if err != nil || parsedRef.Source != ModelSourceCloud {
		return strings.TrimSpace(raw), false
	}

	return parsedRef.Base, true
}

func NormalizePullName(raw string) (string, bool, error) {
	parsedRef, err := ParseRef(raw)
	if err != nil {
		return "", false, err
	}

	if parsedRef.Source != ModelSourceCloud {
		return parsedRef.Base, false, nil
	}

	return toLegacyCloudPullName(parsedRef.Base), true, nil
}

func toLegacyCloudPullName(base string) string {
	if hasExplicitTag(base) {
		return base + "-cloud"
	}

	return base + ":cloud"
}

func hasExplicitTag(name string) bool {
	lastSlash := strings.LastIndex(name, "/")
	lastColon := strings.LastIndex(name, ":")
	return lastColon > lastSlash
}

func parseSourceSuffix(raw string) (string, ModelSource, bool) {
	idx := strings.LastIndex(raw, ":")
	if idx >= 0 {
		suffixRaw := strings.TrimSpace(raw[idx+1:])
		suffix := strings.ToLower(suffixRaw)

		switch suffix {
		case "cloud":
			return raw[:idx], ModelSourceCloud, true
		case "local":
			return raw[:idx], ModelSourceLocal, true
		case "kilocode":
			return raw[:idx], ModelSourceKilocode, true
		}

		if !strings.Contains(suffixRaw, "/") && strings.HasSuffix(suffix, "-cloud") {
			return raw[:idx+1] + suffixRaw[:len(suffixRaw)-len("-cloud")], ModelSourceCloud, true
		}
	}

	return raw, ModelSourceUnspecified, false
}
