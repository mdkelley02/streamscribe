package extractor

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// NewYtDlp returns an extractor backed by yt-dlp, which supports 1000+ platforms
// including YouTube, Twitch, Vimeo, SoundCloud, Kick, X Spaces, and more.
// See https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md for the
// full list. Requires yt-dlp to be installed.
//
// The extractor downloads the full audio track to a temp file via yt-dlp's
// own HTTP client (which works around YouTube's per-stream throttling) and
// then hands the local file to ffmpeg for seeking and decoding. Streaming
// the YouTube CDN URL directly into ffmpeg is roughly 2× realtime due to
// upstream throttling; downloading first is typically 10× realtime or more.
func NewYtDlp() ChunkExtractor {
	e := &ytdlpExtractor{}
	e.ffmpegExtractor.resolveURL = e.resolveURL
	return e
}

// ytdlpExtractor wraps ffmpegExtractor with a yt-dlp-aware resolver and a
// duration probe that avoids re-downloading the media.
type ytdlpExtractor struct {
	ffmpegExtractor
}

// resolveURL downloads the bestaudio track to a temp file and returns the
// file path plus a cleanup that removes it. ffmpeg then seeks and decodes
// from local disk, which is effectively free.
func (y *ytdlpExtractor) resolveURL(
	ctx context.Context,
	url string,
) (string, func(), error) {
	f, err := os.CreateTemp("", "streamscribe-ytdlp-*.audio")
	if err != nil {
		return "", nil, fmt.Errorf("temp file: %w", err)
	}
	tmpPath := f.Name()
	// yt-dlp writes the file itself; close our handle and let it overwrite.
	_ = f.Close()

	cleanup := func() { _ = os.Remove(tmpPath) }

	cmd := exec.CommandContext(ctx, "yt-dlp",
		"-f", "bestaudio",
		"-o", tmpPath,
		"--no-progress",
		"--quiet",
		"--no-part",
		"--force-overwrites",
		url,
	)
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf(
			"yt-dlp download: %w (stderr: %s)",
			err,
			strings.TrimSpace(errBuf.String()),
		)
	}

	return tmpPath, cleanup, nil
}

// GetDuration queries yt-dlp's metadata path instead of downloading the
// media. Overrides ffmpegExtractor.GetDuration, which would otherwise
// trigger a full download just to run ffprobe on the result.
func (y *ytdlpExtractor) GetDuration(
	ctx context.Context,
	url string,
) (time.Duration, error) {
	cmd := exec.CommandContext(ctx, "yt-dlp", "--print", "duration", url)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf(
			"yt-dlp duration: %w (stderr: %s)",
			err,
			strings.TrimSpace(errBuf.String()),
		)
	}

	s := strings.TrimSpace(outBuf.String())
	if s == "" || s == "NA" {
		return 0, fmt.Errorf("yt-dlp returned no duration")
	}

	secs, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse yt-dlp duration %q: %w", s, err)
	}
	return time.Duration(secs * float64(time.Second)), nil
}
