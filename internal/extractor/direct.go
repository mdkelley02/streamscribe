package extractor

import (
	"context"
	"fmt"
	"os"
)

// NewLocalFile returns an extractor for a local audio or video file.
// The path is validated to exist before being passed to ffmpeg.
// No external tools required.
func NewLocalFile() ChunkExtractor {
	return &ffmpegExtractor{
		resolveURL: func(_ context.Context, path string) (string, error) {
			if _, err := os.Stat(path); err != nil {
				return "", fmt.Errorf("local file: %w", err)
			}
			return path, nil
		},
	}
}

// NewDirectURL returns an extractor for any URL that ffmpeg can open directly:
// HTTP/HTTPS media files, HLS manifests, DASH manifests, RTMP streams, etc.
// No URL resolution is performed — the URL is passed to ffmpeg as-is.
// No external tools required.
func NewDirectURL() ChunkExtractor {
	return &ffmpegExtractor{
		resolveURL: func(_ context.Context, url string) (string, error) {
			return url, nil
		},
	}
}

// NewCustom builds an extractor from a user-supplied resolver function.
// The resolver receives the URL string and must return a direct media URL
// that ffmpeg can open (HLS manifest, MP4, RTMP stream, etc.).
func NewCustom(resolve func(context.Context, string) (string, error)) ChunkExtractor {
	return &ffmpegExtractor{resolveURL: resolve}
}
