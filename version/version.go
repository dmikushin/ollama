package version

import (
	"os/exec"
	"strings"
)

// fallbackVersion generates a version string from git when ldflags did not
// inject one. Format: 0.19.0+sha0f31020
func fallbackVersion() string {
	out, err := exec.Command("git", "describe", "--tags", "--first-parent", "--abbrev=7", "--long", "--dirty", "--always").CombinedOutput()
	if err == nil {
		desc := strings.TrimSpace(string(out))
		desc = strings.TrimPrefix(desc, "v")
		// "0.19.0-25-g0f31020-dirty" -> "0.19.0+sha0f31020"
		parts := strings.SplitN(desc, "-", 3)
		if len(parts) == 3 {
			shaRaw := parts[2]
			dirty := ""
			if strings.HasSuffix(shaRaw, "-dirty") {
				dirty = "-dirty"
				shaRaw = strings.TrimSuffix(shaRaw, "-dirty")
			}
			// Remove leading 'g' from g0f31020
			sha := strings.TrimPrefix(shaRaw, "g")
			return parts[0] + "+sha" + sha + dirty
		}
		// Bare commit: "0f31020-dirty"
		dirty := ""
		if strings.HasSuffix(desc, "-dirty") {
			dirty = "-dirty"
			desc = strings.TrimSuffix(desc, "-dirty")
		}
		return desc + "+sha" + desc + dirty
	}
	return "0.0.0+shaunknown"
}

// Version is the version string. Build scripts inject this via ldflags.
// When built manually (go build .) without -ldflags, the init function
// detects the sentinel value and computes a git-SHA-based version.
var Version = "0.0.0+sha"

func init() {
	if Version == "0.0.0+sha" {
		Version = fallbackVersion()
	}
}
