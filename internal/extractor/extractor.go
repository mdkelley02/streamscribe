package extractor

import (
	"context"
	"time"
)

// AudioChunk is a 30-second (or shorter, for the final window) slice of
// 16 kHz mono PCM audio along with its absolute offset within the media.
type AudioChunk struct {
	// Samples holds up to ChunkSamples float32 values. whisper timestamps
	// returned from Process() are relative to sample[0], which corresponds
	// to absolute time Offset within the original stream.
	Samples []float32

	// Offset is the absolute position of sample[0] inside the media.
	// Add this to any whisper segment timestamp to get the media-relative time.
	Offset time.Duration

	// IsFinalChunk is true for the final chunk. The transcriber uses this to suppress
	// the overlap-zone cutoff filter so all remaining segments are emitted.
	IsFinalChunk bool
}

// ChunkExtractor streams audio from a media source in whisper-compatible 30-second
// windows, each overlapping the previous by OverlapSeconds seconds.
type ChunkExtractor interface {
	ExtractChunks(
		ctx context.Context,
		URL string,
		start, end time.Duration,
	) (<-chan AudioChunk, <-chan error)
	GetDuration(ctx context.Context, URL string) (time.Duration, error)
}
