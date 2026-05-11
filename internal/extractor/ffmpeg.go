package extractor

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	SampleRate = 16000

	// whisper.cpp processes exactly 30 seconds of audio per call (3000 mel frames,
	// 1500 after 2x downsampling through the convolutional front-end). Feeding more
	// silently truncates; feeding less pads with silence and wastes encoder cycles.
	ChunkSeconds = 30

	// Advance 28 seconds between windows, keeping a 2-second overlap. The overlap
	// lets the decoder see context for words that straddle chunk boundaries. Each
	// window is solely responsible for emitting segments whose start time falls
	// within [windowOffset, windowOffset+StepSeconds).
	StepSeconds    = 28
	OverlapSeconds = ChunkSeconds - StepSeconds // 2

	ChunkSamples   = ChunkSeconds * SampleRate   // 480,000 float32 samples
	StepSamples    = StepSeconds * SampleRate    // 448,000 float32 samples
	OverlapSamples = OverlapSeconds * SampleRate // 32,000 float32 samples

	bytesPerSample = 4 // float32
	chunkBytes     = ChunkSamples * bytesPerSample
	stepBytes      = StepSamples * bytesPerSample
)

// rawToFloat32 converts a little-endian float32 byte slice to []float32.
func rawToFloat32(b []byte) []float32 {
	n := len(b) / bytesPerSample
	out := make([]float32, n)
	r := bytes.NewReader(b)
	_ = binary.Read(r, binary.LittleEndian, &out)
	return out
}

var _ ChunkExtractor = (*ffmpegExtractor)(nil)

// ffmpegExtractor is the shared implementation of ChunkExtractor for any
// platform that can provide a direct media URL. Callers supply a resolveURL
// function that translates a user-facing URL (e.g. a Twitch or YouTube page
// URL) into a raw media URL suitable for ffmpeg.
//
// Adding support for a new platform requires only a constructor and a
// resolveURL function — no changes to the chunking pipeline.
//
// resolveURL may optionally return a cleanup function (e.g. to delete a
// downloaded temp file). cleanup is invoked after the chunker exits; it
// may be nil when no cleanup is needed.
type ffmpegExtractor struct {
	resolveURL func(ctx context.Context, url string) (mediaPath string, cleanup func(), err error)
}

// ExtractChunks resolves the media URL, launches a single ffmpeg process, and
// produces a stream of AudioChunk values ready to hand directly to whisper.
//
// Pipeline: ffmpeg stdout → pipe → chunker goroutine → chunkCh → transcriber
//
// Both channels are closed when the goroutine exits. Callers must drain
// chunkCh or cancel ctx to avoid goroutine leaks.
func (f *ffmpegExtractor) ExtractChunks(
	ctx context.Context,
	URL string,
	start, end time.Duration,
) (<-chan AudioChunk, <-chan error) {
	// buffer 2 chunks so ffmpeg/whisper can run concurrently
	chunkCh := make(chan AudioChunk, 2)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		mediaURL, cleanup, err := f.resolveURL(ctx, URL)
		if err != nil {
			errCh <- fmt.Errorf("resolve url: %w", err)
			return
		}
		if cleanup != nil {
			defer cleanup()
		}

		args := []string{
			"-hide_banner", "-loglevel", "error",
			// Seek before opening the input — avoids decoding from the start.
			"-ss", fmt.Sprintf("%f", start.Seconds()),
			"-i", mediaURL,
		}
		// end <= 0 means "transcribe to EOF". Omitting -t lets ffmpeg drain
		// the entire input — the chunker terminates on io.EOF from the pipe.
		if end > start {
			duration := end - start
			args = append(args, "-t", fmt.Sprintf("%f", duration.Seconds()))
		}
		args = append(args,
			"-ar", "16000",
			"-ac", "1",
			"-c:a", "pcm_f32le",
			"-f", "f32le",
			"pipe:1",
		)

		cmd := exec.CommandContext(ctx, "ffmpeg", args...)
		var errBuf bytes.Buffer
		cmd.Stderr = &errBuf

		pipe, err := cmd.StdoutPipe()
		if err != nil {
			errCh <- fmt.Errorf("stdout pipe: %w", err)
			return
		}
		if err := cmd.Start(); err != nil {
			errCh <- fmt.Errorf("ffmpeg start: %w", err)
			return
		}

		// prevSamples is the tail of the previous chunk used as overlap context
		// for the next whisper window.
		var prevSamples []float32
		chunkIdx := 0

		for {
			// Window 0 reads a full 30 s to fill the encoder context.
			// Subsequent windows read only 28 s of new audio and prepend prevSamples.
			toRead := stepBytes
			if chunkIdx == 0 {
				toRead = chunkBytes
			}

			raw := make([]byte, toRead)
			n, readErr := io.ReadFull(pipe, raw)
			isLast := readErr == io.ErrUnexpectedEOF || readErr == io.EOF

			if n == 0 {
				// EOF with no new bytes: emit the tail overlap as a final chunk
				// if there is audio that was filtered out by the previous window.
				if len(prevSamples) > 0 {
					offset := start + time.Duration(chunkIdx)*StepSeconds*time.Second
					select {
					case chunkCh <- AudioChunk{Samples: prevSamples, Offset: offset, IsFinalChunk: true}:
					case <-ctx.Done():
					}
				}
				break
			}

			newSamples := rawToFloat32(raw[:n])

			// Build the full 30-second window: overlap tail + fresh audio.
			var windowSamples []float32
			if chunkIdx > 0 && len(prevSamples) > 0 {
				windowSamples = make([]float32, len(prevSamples)+len(newSamples))
				copy(windowSamples, prevSamples)
				copy(windowSamples[len(prevSamples):], newSamples)
			} else {
				windowSamples = newSamples
			}

			// Offset of sample[0] in this window relative to the VOD start.
			// chunkIdx 0 → start+0, chunkIdx 1 → start+28s, chunkIdx 2 → start+56s …
			offset := start + time.Duration(chunkIdx)*StepSeconds*time.Second

			select {
			case chunkCh <- AudioChunk{Samples: windowSamples, Offset: offset, IsFinalChunk: isLast}:
			case <-ctx.Done():
				_ = cmd.Process.Kill()
				return
			}

			// Keep the last OverlapSamples for the next window.
			if len(windowSamples) >= OverlapSamples {
				prevSamples = make([]float32, OverlapSamples)
				copy(prevSamples, windowSamples[len(windowSamples)-OverlapSamples:])
			} else {
				prevSamples = make([]float32, len(windowSamples))
				copy(prevSamples, windowSamples)
			}

			chunkIdx++

			if isLast {
				break
			}
			if readErr != nil && !isLast {
				errCh <- fmt.Errorf("pipe read: %w", readErr)
				_ = cmd.Process.Kill()
				return
			}
		}

		if err := cmd.Wait(); err != nil && ctx.Err() == nil {
			// ffmpeg exits non-zero when killed via context cancel — ignore those.
			errCh <- fmt.Errorf("ffmpeg wait: %w (stderr: %s)", err, strings.TrimSpace(errBuf.String()))
		}
	}()

	return chunkCh, errCh
}

func (f *ffmpegExtractor) GetDuration(ctx context.Context, url string) (time.Duration, error) {
	mediaURL, cleanup, err := f.resolveURL(ctx, url)
	if err != nil {
		return 0, fmt.Errorf("resolve url: %w", err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "csv=p=0",
		"-i", mediaURL,
	)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("ffprobe: %w (stderr: %s)", err, strings.TrimSpace(errBuf.String()))
	}

	cmdOutputString := strings.TrimSpace(outBuf.String())
	seconds, err := strconv.ParseFloat(cmdOutputString, 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration: %w (output: %s)", err, cmdOutputString)
	}

	return time.Duration(seconds * float64(time.Second)), nil
}
