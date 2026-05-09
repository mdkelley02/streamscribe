// Package streamscribe transcribes audio from any media platform supported by
// yt-dlp, as well as podcast RSS feeds, local files, and raw media URLs.
//
// Basic usage:
//
//	s, err := streamscribe.New("models/ggml-large-v3-turbo.bin")
//	if err != nil { ... }
//	defer s.Close()
//
//	segments, errCh := s.Transcribe(ctx, "https://www.youtube.com/watch?v=...")
//	for seg := range segments {
//	    fmt.Printf("[%s] %s\n", seg.Start, seg.Text)
//	}
//	if err := <-errCh; err != nil { ... }
package streamscribe

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync/atomic"
	"time"

	"github.com/mdkelley02/streamscribe/internal/extractor"
	"github.com/mdkelley02/streamscribe/internal/transcriber"
)

type StreamScriber interface {
	Transcribe(
		ctx context.Context,
		url string,
		opts ...TranscribeOption,
	) (<-chan Segment, <-chan error)
	TranscribeToFile(
		ctx context.Context,
		url string,
		outputPath string,
		opts ...TranscribeOption,
	) error
	Close() error
}

// Segment is a single transcribed utterance with media-relative timestamps.
type Segment = transcriber.Segment

// ── Progress events ───────────────────────────────────────────────────────────

// ProgressStage labels a phase of a transcription job. New stages may be
// added in the future; consumers should ignore unknown stages.
type ProgressStage string

const (
	// StageResolving is emitted once before URL resolution starts.
	StageResolving ProgressStage = "resolving"
	// StageDownloadingStarted is emitted when the first PCM chunk lands —
	// i.e. ffmpeg has opened the input and begun streaming bytes.
	StageDownloadingStarted ProgressStage = "downloading_started"
	// StageChunkTranscribed is emitted after each chunk's segments are
	// flushed. ChunkIdx is 0-based; ChunkTotal is 0 when unknown.
	StageChunkTranscribed ProgressStage = "chunk_transcribed"
	// StageDone is emitted once on successful completion before the channel
	// closes. Not emitted on error or cancellation.
	StageDone ProgressStage = "done"
)

// Progress is a single progress event in the lifecycle of a Transcribe call.
type Progress struct {
	Stage      ProgressStage `json:"stage"`
	ChunkIdx   int           `json:"chunk_idx,omitempty"`
	ChunkTotal int           `json:"chunk_total,omitempty"`
	Elapsed    time.Duration `json:"elapsed"`
}

// StreamScribe loads a whisper model and transcribes audio from media sources.
// Create one with New; reuse it across multiple Transcribe calls.
//
// Concurrency: a single StreamScribe serializes concurrent Transcribe calls
// internally — they queue on a model-level mutex and run one at a time. The
// GPU is the bottleneck for whisper.cpp anyway. Pool StreamScribe instances
// (typically one per model size) if you need true parallelism.
type StreamScribe struct {
	log         *slog.Logger
	transcriber transcriber.StreamTranscriber
	extractor   extractor.ChunkExtractor

	defaultChunk transcriber.ChunkOptions
}

// Close releases the whisper model instances held by this StreamScribe.
func (tr *StreamScribe) Close() error {
	if err := tr.transcriber.Close(); err != nil {
		return fmt.Errorf("failed to close transcriber: %w", err)
	}
	return nil
}

// ── Constructor ───────────────────────────────────────────────────────────────

// Option configures a StreamScribe.
type Option func(*streamScribeConfig)

type streamScribeConfig struct {
	workers   int
	threads   int
	extractor extractor.ChunkExtractor
	logger    *slog.Logger

	// defaults applied to every Transcribe call unless overridden.
	defaultChunk transcriber.ChunkOptions
}

// WithWorkers sets the number of parallel whisper model instances.
// Defaults to 1. Keep at 1 for CUDA — the GPU handles parallelism internally
// and multiple instances fragment VRAM.
func WithWorkers(n int) Option {
	return func(c *streamScribeConfig) { c.workers = n }
}

// WithThreads sets the CPU threads per whisper worker.
// Defaults to 4. With CUDA this only affects mel-spectrogram pre-processing.
func WithThreads(n int) Option {
	return func(c *streamScribeConfig) { c.threads = n }
}

// WithLogger sets the structured logger used for progress output.
// Defaults to slog.Default().
func WithLogger(l *slog.Logger) Option {
	return func(c *streamScribeConfig) { c.logger = l }
}

// WithYtDlp configures the StreamScribe to resolve media URLs via yt-dlp,
// supporting 1000+ platforms including YouTube, Twitch, Vimeo, SoundCloud,
// Kick, X Spaces, and more. This is the default.
// See https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
func WithYtDlp() Option {
	return func(c *streamScribeConfig) { c.extractor = extractor.NewYtDlp() }
}

// WithPodcastRSS configures the StreamScribe to accept podcast RSS feed URLs.
// The URL passed to Transcribe must be the RSS feed URL; the most recent
// episode's audio enclosure is fetched and transcribed. No external tools needed.
func WithPodcastRSS() Option {
	return func(c *streamScribeConfig) { c.extractor = extractor.NewPodcastRSS() }
}

// WithLocalFile configures the StreamScribe to read audio from local file paths.
// No external tools needed.
func WithLocalFile() Option {
	return func(c *streamScribeConfig) { c.extractor = extractor.NewLocalFile() }
}

// WithDirectURL configures the StreamScribe to pass URLs directly to ffmpeg
// without any resolution step. Suitable for raw HTTP/HTTPS media files,
// HLS manifests, DASH manifests, RTMP streams, and any other URL scheme
// ffmpeg can open natively. No external tools needed.
func WithDirectURL() Option {
	return func(c *streamScribeConfig) { c.extractor = extractor.NewDirectURL() }
}

// WithCustomSource configures the StreamScribe with a user-supplied URL resolver.
// The resolve function receives the URL string passed to Transcribe and must
// return a direct media URL that ffmpeg can open (HLS manifest, MP4, etc.).
//
// Example — resolving via a hypothetical internal CDN tool:
//
//	streamscribe.WithCustomSource(func(ctx context.Context, url string) (string, error) {
//	    return mycdn.ResolveAudioURL(ctx, url)
//	})
func WithCustomSource(resolve func(context.Context, string) (string, error)) Option {
	return func(c *streamScribeConfig) { c.extractor = extractor.NewCustom(resolve) }
}

// WithDefaultLanguage sets the default whisper language hint applied to every
// Transcribe call (e.g. "en", "es", or "auto"). Per-call WithLanguage wins.
func WithDefaultLanguage(lang string) Option {
	return func(c *streamScribeConfig) { c.defaultChunk.Language = lang }
}

// WithDefaultVAD enables voice-activity detection on every Transcribe call,
// using the silero VAD model at modelPath. Per-call WithVAD / WithoutVAD wins.
//
// Recommended for short clips (Reels/Shorts) where leading or trailing silence
// is common — the encoder skips silent regions entirely. Has a small cost on
// long-form content where speech density is already high.
//
// modelPath should point to a silero ggml-bundled VAD model
// (e.g. "whisper.cpp/models/ggml-silero-v5.1.2.bin"). If the path is empty,
// VAD is not enabled.
func WithDefaultVAD(modelPath string) Option {
	return func(c *streamScribeConfig) {
		if modelPath == "" {
			return
		}
		c.defaultChunk.VAD = true
		c.defaultChunk.VADModelPath = modelPath
	}
}

// New loads the whisper model at modelPath and returns a StreamScribe ready for use.
// modelPath should point to a whisper .bin file
// (e.g. "whisper.cpp/models/ggml-large-v3-turbo.bin").
//
// By default the StreamScribe uses yt-dlp for URL resolution; override with
// WithPodcastRSS, WithLocalFile, WithDirectURL, or WithCustomSource.
func New(modelPath string, opts ...Option) (StreamScriber, error) {
	cfg := &streamScribeConfig{
		workers:   1,
		threads:   4,
		extractor: extractor.NewYtDlp(),
		logger:    slog.Default(),
	}
	for _, o := range opts {
		o(cfg)
	}

	t, err := transcriber.New(transcriber.Options{
		ModelPath:        modelPath,
		NumWorkers:       cfg.workers,
		ThreadsPerWorker: cfg.threads,
	})
	if err != nil {
		return nil, err
	}

	return &StreamScribe{
		log:          cfg.logger,
		transcriber:  t,
		extractor:    cfg.extractor,
		defaultChunk: cfg.defaultChunk,
	}, nil
}

// ── Transcribe ────────────────────────────────────────────────────────────────

// TranscribeOption configures a single Transcribe or TranscribeToFile call.
type TranscribeOption func(*transcribeConfig)

type transcribeConfig struct {
	start time.Duration
	end   time.Duration

	chunk     transcriber.ChunkOptions
	chunkSet  chunkSetMask
	progressC chan<- Progress
}

// chunkSetMask tracks which chunk fields the caller has explicitly set, so we
// know when to override the StreamScribe defaults vs leave them alone.
type chunkSetMask struct {
	threads       bool
	language      bool
	translate     bool
	beamSize      bool
	temperature   bool
	initialPrompt bool
	vad           bool
}

// WithStart sets the start offset within the media source. Defaults to 0.
func WithStart(d time.Duration) TranscribeOption {
	return func(c *transcribeConfig) { c.start = d }
}

// WithEnd sets the end offset within the media source.
// If not specified, transcription runs until the source ends.
func WithEnd(d time.Duration) TranscribeOption {
	return func(c *transcribeConfig) { c.end = d }
}

// WithLanguage sets the whisper language hint for this call.
// Use "" or "auto" to auto-detect.
func WithLanguage(lang string) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.Language = lang
		c.chunkSet.language = true
	}
}

// WithTranslate causes whisper to translate non-English audio into English.
func WithTranslate(translate bool) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.Translate = translate
		c.chunkSet.translate = true
	}
}

// WithBeamSize selects beam search with the given beam width. 0 or 1 means
// greedy. Note: the bundled Go bindings hard-code SAMPLING_GREEDY in
// NewContext, so this knob may not switch sampling strategy until the binding
// is patched. The parameter is set unconditionally for forward compatibility.
func WithBeamSize(n int) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.BeamSize = n
		c.chunkSet.beamSize = true
	}
}

// WithTemperature sets the whisper sampling temperature. 0 = greedy.
func WithTemperature(t float32) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.Temperature = t
		c.chunkSet.temperature = true
	}
}

// WithInitialPrompt seeds whisper with the given text on each worker's first
// chunk. Subsequent chunks chain off their own emitted text.
func WithInitialPrompt(prompt string) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.InitialPrompt = prompt
		c.chunkSet.initialPrompt = true
	}
}

// WithRequestThreads overrides the StreamScribe's thread count for this call.
func WithRequestThreads(n int) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.Threads = n
		c.chunkSet.threads = true
	}
}

// VADOption tunes voice-activity detection.
type VADOption func(*transcriber.ChunkOptions)

// VADThreshold sets the speech-probability cutoff (0..1). 0 = library default.
func VADThreshold(t float32) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADThreshold = t }
}

// VADMinSpeechMs sets the minimum speech segment length in ms.
func VADMinSpeechMs(ms int) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADMinSpeechMs = ms }
}

// VADMinSilenceMs sets the minimum silence segment length in ms.
func VADMinSilenceMs(ms int) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADMinSilenceMs = ms }
}

// VADMaxSpeechSec caps a single speech segment in seconds.
func VADMaxSpeechSec(s float32) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADMaxSpeechSec = s }
}

// VADSpeechPadMs adds padding ms around detected speech.
func VADSpeechPadMs(ms int) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADSpeechPadMs = ms }
}

// VADSamplesOverlapSec sets the inter-segment overlap in seconds.
func VADSamplesOverlapSec(sec float32) VADOption {
	return func(c *transcriber.ChunkOptions) { c.VADSamplesOverlapSec = sec }
}

// WithVAD enables voice-activity detection for this call. modelPath must
// point to a silero VAD model. Pass VADOption tuners for thresholds.
func WithVAD(modelPath string, vadOpts ...VADOption) TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.VAD = true
		c.chunk.VADModelPath = modelPath
		for _, o := range vadOpts {
			o(&c.chunk)
		}
		c.chunkSet.vad = true
	}
}

// WithoutVAD disables voice-activity detection for this call, even if the
// StreamScribe was constructed with WithDefaultVAD.
func WithoutVAD() TranscribeOption {
	return func(c *transcribeConfig) {
		c.chunk.VAD = false
		c.chunk.VADModelPath = ""
		c.chunkSet.vad = true
	}
}

// WithProgress streams Progress events for this call to ch. The channel is
// not closed by streamscribe — the caller owns it and must drain or buffer
// generously to avoid blocking the pipeline. Sends respect ctx cancellation.
func WithProgress(ch chan<- Progress) TranscribeOption {
	return func(c *transcribeConfig) { c.progressC = ch }
}

// Transcribe transcribes the media at url and streams segments as they are
// produced. Both returned channels are closed when transcription completes
// or ctx is cancelled.
//
// Callers must drain the segment channel or cancel ctx to avoid goroutine leaks.
func (tr *StreamScribe) Transcribe(
	ctx context.Context,
	url string,
	opts ...TranscribeOption,
) (<-chan Segment, <-chan error) {
	cfg := &transcribeConfig{}
	for _, o := range opts {
		o(cfg)
	}

	chunkOpts := tr.resolveChunkOptions(cfg)

	outSegCh := make(chan transcriber.Segment, 128)
	outErrCh := make(chan error, 1)

	go func() {
		defer close(outSegCh)
		defer close(outErrCh)

		// innerCtx lets defer cancel both pipeline goroutines when this
		// orchestrator exits — prevents extractor goroutine leaks if the
		// transcriber errors and stops consuming chunkCh.
		innerCtx, innerCancel := context.WithCancel(ctx)
		defer innerCancel()

		log := tr.log.With("url", url)
		wallStart := time.Now()

		emit := func(p Progress) {
			if cfg.progressC == nil {
				return
			}
			p.Elapsed = time.Since(wallStart)
			select {
			case cfg.progressC <- p:
			case <-innerCtx.Done():
			}
		}

		emit(Progress{Stage: StageResolving})

		// chunkTotal is filled by GetDuration in the background when end is
		// unset, so progress events for early chunks may carry total=0
		// (unknown). atomic guards the cross-goroutine read/write.
		var chunkTotal atomic.Int64

		windowDuration := time.Duration(0)
		if cfg.end > cfg.start {
			windowDuration = cfg.end - cfg.start
			chunkTotal.Store(int64(calculateEstimatedChunks(windowDuration)))
		}

		// Resolve duration in the background only when we don't already know
		// it. Failure is non-fatal — it just means progress events lack a
		// chunk total. Don't gate transcription on this.
		if cfg.end <= cfg.start {
			go func() {
				dur, err := tr.extractor.GetDuration(innerCtx, url)
				if err != nil {
					log.Debug("background duration probe failed", "err", err)
					return
				}
				if dur > cfg.start {
					chunkTotal.Store(int64(calculateEstimatedChunks(dur - cfg.start)))
				}
			}()
		}

		log.Info(
			"starting transcription",
			"chunk_size", extractor.ChunkSeconds*time.Second,
			"start", cfg.start,
			"end_known", windowDuration > 0,
			"estimated_chunks", chunkTotal.Load(),
		)

		chunkCh, extractErrCh := tr.extractor.ExtractChunks(innerCtx, url, cfg.start, cfg.end)

		// Wrap chunkCh to fire StageDownloadingStarted when the first chunk lands.
		hookedChunkCh := make(chan extractor.AudioChunk, 1)
		go func() {
			defer close(hookedChunkCh)
			first := true
			for c := range chunkCh {
				if first {
					emit(Progress{Stage: StageDownloadingStarted})
					first = false
				}
				select {
				case hookedChunkCh <- c:
				case <-innerCtx.Done():
					return
				}
			}
		}()

		chunkOpts.OnChunkDone = func(idx int) {
			emit(Progress{
				Stage:      StageChunkTranscribed,
				ChunkIdx:   idx,
				ChunkTotal: int(chunkTotal.Load()),
			})
		}

		segCh, transcribeErrCh := tr.transcriber.TranscribeChunks(innerCtx, hookedChunkCh, chunkOpts)

		segmentIndex := 0
		for seg := range segCh {
			select {
			case outSegCh <- seg:
				log.Debug("transcribed segment",
					"segment_index", segmentIndex,
					"text_len", len(seg.Text),
					"elapsed", time.Since(wallStart).Round(time.Second),
				)
				segmentIndex++
			case <-innerCtx.Done():
				outErrCh <- innerCtx.Err()
				return
			}
		}

		if err := <-transcribeErrCh; err != nil {
			outErrCh <- fmt.Errorf("transcriber: %w", err)
			return
		}
		if err := <-extractErrCh; err != nil {
			outErrCh <- fmt.Errorf("extractor: %w", err)
			return
		}

		emit(Progress{Stage: StageDone})

		log.Info("transcription complete",
			"total_segments", segmentIndex,
			"elapsed", time.Since(wallStart).Round(time.Second),
		)
	}()

	return outSegCh, outErrCh
}

// resolveChunkOptions merges StreamScribe defaults with per-call overrides.
func (tr *StreamScribe) resolveChunkOptions(cfg *transcribeConfig) transcriber.ChunkOptions {
	out := tr.defaultChunk
	set := cfg.chunkSet
	src := cfg.chunk

	if set.threads {
		out.Threads = src.Threads
	}
	if set.language {
		out.Language = src.Language
	}
	if set.translate {
		out.Translate = src.Translate
	}
	if set.beamSize {
		out.BeamSize = src.BeamSize
	}
	if set.temperature {
		out.Temperature = src.Temperature
	}
	if set.initialPrompt {
		out.InitialPrompt = src.InitialPrompt
	}
	if set.vad {
		out.VAD = src.VAD
		out.VADModelPath = src.VADModelPath
		out.VADThreshold = src.VADThreshold
		out.VADMinSpeechMs = src.VADMinSpeechMs
		out.VADMinSilenceMs = src.VADMinSilenceMs
		out.VADMaxSpeechSec = src.VADMaxSpeechSec
		out.VADSpeechPadMs = src.VADSpeechPadMs
		out.VADSamplesOverlapSec = src.VADSamplesOverlapSec
	}
	return out
}

// TranscribeToFile transcribes the media at url and writes the result to
// outputPath as a JSON array of Segment objects.
func (tr *StreamScribe) TranscribeToFile(
	ctx context.Context,
	url string,
	outputPath string,
	opts ...TranscribeOption,
) error {
	segCh, errCh := tr.Transcribe(ctx, url, opts...)

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer f.Close()

	fmt.Fprintln(f, "[")
	jsonEncoder := json.NewEncoder(f)
	jsonEncoder.SetIndent("", "  ")

	first := true
	for seg := range segCh {
		if !first {
			fmt.Fprint(f, ",\n")
		}
		if err := jsonEncoder.Encode(seg); err != nil {
			return fmt.Errorf("failed to encode segment: %w", err)
		}
		first = false
	}

	if err := <-errCh; err != nil {
		return fmt.Errorf("transcription error: %w", err)
	}

	fmt.Fprintln(f, "]")
	return nil
}

func calculateEstimatedChunks(windowDuration time.Duration) int {
	if windowDuration <= extractor.StepSeconds*time.Second {
		return 1
	}

	return int(
		math.Ceil(float64(windowDuration)/float64(extractor.StepSeconds*time.Second)),
	) + 1
}

func Drain[T any](in <-chan T, errCh <-chan error) ([]T, error) {
	var out []T
	for item := range in {
		out = append(out, item)
	}
	if err := <-errCh; err != nil {
		return nil, err
	}
	return out, nil
}
