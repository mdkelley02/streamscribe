package transcriber

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/mdkelley02/streamscribe/internal/extractor"
)

// Segment is a single transcribed utterance with VOD-relative timestamps.
type Segment struct {
	Start time.Duration `json:"start"`
	End   time.Duration `json:"end"`
	Text  string        `json:"text"`
}

// Options controls the transcription worker pool.
type Options struct {
	// ModelPath is the path to the whisper .bin file.
	// Example: "whisper.cpp/models/ggml-tiny.en.bin"
	ModelPath string

	// NumWorkers is the number of independent model instances loaded in parallel.
	// Each worker has its own whisper context and its own initialPrompt chain,
	// so continuity is maintained within each worker's chunk subsequence.
	// Keep at 1 for CUDA — the GPU handles parallelism internally and multiple
	// instances fragment VRAM.
	NumWorkers int

	// ThreadsPerWorker is the n_threads value passed to each whisper context
	// when ChunkOptions.Threads is unset. With CUDA this only affects
	// mel-spectrogram pre-processing (CPU side); 4 is a sane default.
	ThreadsPerWorker int
}

// ChunkOptions tunes a single TranscribeChunks invocation. Zero values mean
// "use whisper.cpp's default for that knob".
type ChunkOptions struct {
	// Threads overrides Options.ThreadsPerWorker for this call. 0 = use the
	// pool default.
	Threads int

	// Language hint for whisper. "" = auto detect (the binding default).
	// Use "en", "es", etc., or "auto" to be explicit.
	Language string

	// Translate causes whisper to translate non-English audio into English.
	Translate bool

	// BeamSize > 1 selects beam search. Note: the bundled Go bindings
	// hard-code SAMPLING_GREEDY in NewContext, so SetBeamSize alone may not
	// switch sampling strategy until the binding is patched (see TODO P1).
	BeamSize int

	// Temperature is whisper's sampling temperature. 0 = greedy (default).
	Temperature float32

	// InitialPrompt seeds each worker's first chunk. Workers continue to
	// chain prompts from their own emitted text after that.
	InitialPrompt string

	// VAD enables voice-activity detection. Requires VADModelPath to be set.
	VAD          bool
	VADModelPath string

	// VAD tuning. 0 values defer to whisper.cpp defaults.
	VADThreshold         float32
	VADMinSpeechMs       int
	VADMinSilenceMs      int
	VADMaxSpeechSec      float32
	VADSpeechPadMs       int
	VADSamplesOverlapSec float32

	// OnChunkDone fires after the merger flushes all segments for chunk idx
	// (0-based). Useful for progress reporting. Called from the merger
	// goroutine; must be cheap and non-blocking.
	OnChunkDone func(idx int)
}

// StreamTranscriber extends Transcriber with a streaming, chunk-aware path
// optimised for long VODs.
type StreamTranscriber interface {
	Close() error
	TranscribeChunks(
		ctx context.Context,
		chunkCh <-chan extractor.AudioChunk,
		opts ChunkOptions,
	) (<-chan Segment, <-chan error)
}

type transcriber struct {
	models []whisper.Model
	opts   Options

	// mu serializes TranscribeChunks calls on this instance. The C whisper
	// contexts owned by each model are not safe to share across concurrent
	// inference calls, so we hold the lock for the full duration of one
	// streaming job. Callers wanting parallelism should pool transcribers.
	mu sync.Mutex
}

// New loads opts.NumWorkers independent copies of the whisper model into memory.
// Each copy is fully independent (its own C whisper_context), allowing safe
// concurrent inference across goroutines.
//
// Memory cost: ~75 MB × NumWorkers for tiny.en, ~460 MB for base.en.
// With 12 GB RAM and tiny.en the default of 7 workers costs ~525 MB.
func New(opts Options) (StreamTranscriber, error) {
	if opts.NumWorkers < 1 {
		opts.NumWorkers = 1
	}
	if opts.ThreadsPerWorker < 1 {
		opts.ThreadsPerWorker = 1
	}

	models := make([]whisper.Model, opts.NumWorkers)
	for i := range models {
		model, err := whisper.New(opts.ModelPath)
		if err != nil {
			// Close any already-loaded models before returning.
			for j := 0; j < i; j++ {
				models[j].Close()
			}

			return nil, fmt.Errorf("load model instance %d: %w", i, err)
		}

		models[i] = model
	}

	return &transcriber{
		models: models,
		opts:   opts,
	}, nil
}

func (w *transcriber) Close() error {
	var closeModelErrors []error
	for _, model := range w.models {
		if err := model.Close(); err != nil {
			closeModelErrors = append(closeModelErrors, err)
		}
	}
	return errors.Join(closeModelErrors...)
}

// ── Worker pool internals ─────────────────────────────────────────────────────

type workerTask struct {
	chunk    extractor.AudioChunk
	chunkIdx int
	resultCh chan workerResult // buffered(1); worker writes exactly one value
}

type workerResult struct {
	segments []Segment
	err      error
}

// TranscribeChunks is the high-throughput streaming path.
//
// Concurrency contract: only one TranscribeChunks call may run on a given
// transcriber instance at a time. Concurrent callers serialize on an internal
// mutex — the GPU is the bottleneck for whisper.cpp anyway, so this is a
// natural choke point. Pool transcribers if you need true parallelism.
//
// Architecture:
//
//	chunkCh (from extractor)
//	  → dispatcher goroutine  (round-robins chunks across N workers)
//	       ↓ N worker channels
//	  → worker goroutines[0..N-1]  (each runs whisper sequentially, maintains its own prompt chain)
//	       ↓ per-chunk result channels (buffered)
//	  → merger goroutine  (reads result channels IN CHUNK ORDER, emits to segCh)
//	       ↓
//	  segCh (VOD-ordered segments)
//
// Ordering guarantee: segments are emitted in ascending timestamp order because
// the merger reads per-chunk result channels in the order the dispatcher created
// them, regardless of which worker finished first.
//
// Prompt continuity: worker K handles chunks 0, N, 2N, ... (every Nth chunk).
// Each worker carries its own initialPrompt across its subsequence. The
// prompt gap between consecutive chunks of the same worker is
// (NumWorkers × StepSeconds) seconds — e.g. 7 workers × 28 s = 196 s.
func (w *transcriber) TranscribeChunks(
	ctx context.Context,
	chunkCh <-chan extractor.AudioChunk,
	opts ChunkOptions,
) (<-chan Segment, <-chan error) {
	numWorkers := len(w.models)

	segCh := make(chan Segment, 128)
	errCh := make(chan error, 1)

	// One task channel per worker. Buffer of 1 lets the dispatcher stay one chunk
	// ahead of each worker without blocking, allowing better pipeline overlap.
	taskChs := make([]chan workerTask, numWorkers)
	for i := range taskChs {
		taskChs[i] = make(chan workerTask, 1)
	}

	// mergerCh carries per-chunk result channels in strict chunk order.
	// The merger reads from it to reconstruct ordered output.
	mergerCh := make(chan chan workerResult, numWorkers*3)

	// Acquire the instance lock asynchronously so TranscribeChunks returns
	// immediately. Concurrent callers queue here while the upstream extractor
	// keeps producing bytes (bounded by chunkCh's buffer + ffmpeg back-pressure).
	// The lock is held until every goroutine launched below has exited; we
	// can't release earlier because CGo Process calls are not ctx-cancelable
	// and a worker may still be inside whisper inference.
	go func() {
		select {
		case <-ctx.Done():
			// Caller cancelled before we got the lock: emit ctx error and don't
			// run any workers / dispatcher / merger. Drain chunkCh so the
			// extractor can clean up.
			errCh <- ctx.Err()
			close(segCh)
			close(errCh)
			for range chunkCh {
			}
			return
		default:
		}
		w.mu.Lock()
		// ctx may have been cancelled while we waited.
		if ctx.Err() != nil {
			w.mu.Unlock()
			errCh <- ctx.Err()
			close(segCh)
			close(errCh)
			for range chunkCh {
			}
			return
		}

		var wg sync.WaitGroup
		w.runPipeline(ctx, &wg, chunkCh, segCh, errCh, taskChs, mergerCh, opts)
		wg.Wait()
		w.mu.Unlock()
	}()

	return segCh, errCh
}

// runPipeline starts the dispatcher, workers, and merger and returns once they
// are all running. The caller must wg.Wait() before releasing the model lock.
func (w *transcriber) runPipeline(
	ctx context.Context,
	wg *sync.WaitGroup,
	chunkCh <-chan extractor.AudioChunk,
	segCh chan Segment,
	errCh chan error,
	taskChs []chan workerTask,
	mergerCh chan chan workerResult,
	opts ChunkOptions,
) {
	numWorkers := len(w.models)

	// ── Workers ───────────────────────────────────────────────────────────────
	wg.Add(numWorkers)
	for i, model := range w.models {
		go func(model whisper.Model, taskCh <-chan workerTask) {
			defer wg.Done()
			w.runWorker(ctx, model, taskCh, opts)
		}(model, taskChs[i])
	}

	// ── Dispatcher ────────────────────────────────────────────────────────────
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			for _, ch := range taskChs {
				close(ch)
			}
			close(mergerCh)
		}()

		chunkIdx := 0
		for chunk := range chunkCh {
			if ctx.Err() != nil {
				return
			}

			resultCh := make(chan workerResult, 1)
			workerIdx := chunkIdx % numWorkers

			// Enqueue to merger FIRST (before the worker runs) so the merger
			// always sees result channels in chunk-index order.
			select {
			case mergerCh <- resultCh:
			case <-ctx.Done():
				return
			}

			// Hand off to the assigned worker. Blocks if the worker's task
			// channel is full (buffer=1), naturally throttling the dispatcher.
			select {
			case taskChs[workerIdx] <- workerTask{chunk: chunk, chunkIdx: chunkIdx, resultCh: resultCh}:
			case <-ctx.Done():
				return
			}

			chunkIdx++
		}
	}()

	// ── Merger ────────────────────────────────────────────────────────────────
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(segCh)
		defer close(errCh)

		flushedIdx := 0
		for resultCh := range mergerCh {
			var result workerResult
			select {
			case result = <-resultCh:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}

			if result.err != nil {
				errCh <- result.err
				return
			}

			for _, seg := range result.segments {
				select {
				case segCh <- seg:
				case <-ctx.Done():
					errCh <- ctx.Err()
					return
				}
			}

			if opts.OnChunkDone != nil {
				opts.OnChunkDone(flushedIdx)
			}
			flushedIdx++
		}
	}()
}

// runWorker is the per-goroutine whisper inference loop for one model instance.
// It processes tasks sequentially, maintaining its own initialPrompt chain
// across the subsequence of chunks assigned to it.
func (w *transcriber) runWorker(
	ctx context.Context,
	model whisper.Model,
	taskCh <-chan workerTask,
	opts ChunkOptions,
) {
	initialPrompt := opts.InitialPrompt
	stepDur := time.Duration(extractor.StepSeconds) * time.Second

	threads := opts.Threads
	if threads <= 0 {
		threads = w.opts.ThreadsPerWorker
	}

	for task := range taskCh {
		if ctx.Err() != nil {
			task.resultCh <- workerResult{err: ctx.Err()}
			return
		}

		segments, err := processChunk(
			model,
			threads,
			task.chunk,
			initialPrompt,
			stepDur,
			opts,
		)
		task.resultCh <- workerResult{segments: segments, err: err}

		if err != nil {
			return
		}

		// Build the next prompt from this chunk's transcript.
		if len(segments) > 0 {
			var sb strings.Builder
			for _, s := range segments {
				sb.WriteString(s.Text)
				sb.WriteByte(' ')
			}
			initialPrompt = lastNWords(sb.String(), 30)
		}
	}
}

// processChunk runs whisper inference on one AudioChunk and returns
// VOD-relative segments with the overlap filter applied.
func processChunk(
	model whisper.Model,
	threads int,
	chunk extractor.AudioChunk,
	initialPrompt string,
	stepDur time.Duration,
	opts ChunkOptions,
) ([]Segment, error) {
	wCtx, err := model.NewContext()
	if err != nil {
		return nil, fmt.Errorf("new whisper context at %s: %w", chunk.Offset, err)
	}

	wCtx.SetThreads(uint(threads))

	if initialPrompt != "" {
		wCtx.SetInitialPrompt(initialPrompt)
	}

	if opts.Language != "" {
		if err := wCtx.SetLanguage(opts.Language); err != nil {
			return nil, fmt.Errorf("SetLanguage(%q) at %s: %w", opts.Language, chunk.Offset, err)
		}
	}
	if opts.Translate {
		wCtx.SetTranslate(true)
	}
	if opts.BeamSize > 0 {
		wCtx.SetBeamSize(opts.BeamSize)
	}
	if opts.Temperature > 0 {
		wCtx.SetTemperature(opts.Temperature)
	}

	if opts.VAD {
		if opts.VADModelPath != "" {
			wCtx.SetVADModelPath(opts.VADModelPath)
		}
		wCtx.SetVAD(true)
		if opts.VADThreshold > 0 {
			wCtx.SetVADThreshold(opts.VADThreshold)
		}
		if opts.VADMinSpeechMs > 0 {
			wCtx.SetVADMinSpeechMs(opts.VADMinSpeechMs)
		}
		if opts.VADMinSilenceMs > 0 {
			wCtx.SetVADMinSilenceMs(opts.VADMinSilenceMs)
		}
		if opts.VADMaxSpeechSec > 0 {
			wCtx.SetVADMaxSpeechSec(opts.VADMaxSpeechSec)
		}
		if opts.VADSpeechPadMs > 0 {
			wCtx.SetVADSpeechPadMs(opts.VADSpeechPadMs)
		}
		if opts.VADSamplesOverlapSec > 0 {
			wCtx.SetVADSamplesOverlap(opts.VADSamplesOverlapSec)
		}
	}

	// For the final (short) chunk, reduce the encoder attention window
	// proportionally. whisper.cpp pads audio to 30 s with silence; SetAudioCtx
	// limits the transformer encoder to only the first N of its 1500 input tokens,
	// skipping silent padding entirely. On GPU this saves meaningful encoder time.
	//
	// 1500 tokens / 30 s = 50 tokens/s → audioCtx = seconds * 50
	if chunk.IsFinalChunk && len(chunk.Samples) < extractor.ChunkSamples {
		audioCtx := uint(len(chunk.Samples)) * 1500 / uint(extractor.ChunkSamples)
		if audioCtx < 64 {
			audioCtx = 64
		}
		wCtx.SetAudioCtx(audioCtx)
	}

	if err := wCtx.Process(chunk.Samples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper.Process at %s: %w", chunk.Offset, err)
	}

	cutoff := chunk.Offset + stepDur
	var segments []Segment

	for {
		seg, err := wCtx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("NextSegment at %s: %w", chunk.Offset, err)
		}

		absStart := chunk.Offset + seg.Start
		absEnd := chunk.Offset + seg.End

		// Emit only segments whose start falls in this chunk's responsibility
		// zone [chunk.Offset, chunk.Offset+stepDur). The last chunk emits all.
		if !chunk.IsFinalChunk && absStart >= cutoff {
			continue
		}
		if !chunk.IsFinalChunk && absEnd > cutoff {
			absEnd = cutoff
		}

		segments = append(segments, Segment{Start: absStart, End: absEnd, Text: seg.Text})
	}

	return segments, nil
}

// lastNWords returns the last n whitespace-separated tokens from s.
func lastNWords(s string, n int) string {
	words := strings.Fields(s)
	if len(words) <= n {
		return strings.Join(words, " ")
	}
	return strings.Join(words[len(words)-n:], " ")
}
