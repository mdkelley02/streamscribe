package extractor

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// NewYtDlp returns an extractor backed by yt-dlp, which supports 1000+ platforms
// including YouTube, Twitch, Vimeo, SoundCloud, Kick, X Spaces, and more.
// See https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md for the
// full list. Requires yt-dlp to be installed.
func NewYtDlp() ChunkExtractor {
	return &ffmpegExtractor{
		resolveURL: func(ctx context.Context, url string) (string, error) {
			cmd := exec.CommandContext(ctx, "yt-dlp", "-f", "bestaudio", "--get-url", url)
			var outBuf, errBuf bytes.Buffer
			cmd.Stdout = &outBuf
			cmd.Stderr = &errBuf

			if err := cmd.Run(); err != nil {
				return "", fmt.Errorf(
					"yt-dlp: %w, stderr: %s",
					err,
					strings.TrimSpace(errBuf.String()),
				)
			}

			// yt-dlp may return multiple lines for DASH formats; take the first URL.
			resolved := strings.SplitN(strings.TrimSpace(outBuf.String()), "\n", 2)[0]
			if resolved == "" {
				return "", fmt.Errorf("yt-dlp returned empty URL")
			}
			return resolved, nil
		},
	}
}
