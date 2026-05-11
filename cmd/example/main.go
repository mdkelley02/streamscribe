package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mdkelley02/streamscribe"
)

func main() {
	var (
		modelPath = flag.String(
			"model",
			"whisper.cpp/models/ggml-large-v3-turbo.bin",
			"path to whisper .bin model",
		)
		mediaURL = flag.String("url", "", "media URL (required)")
		startSec = flag.Float64("start", 0, "start offset in seconds")
		endSec   = flag.Float64("end", 0, "end offset in seconds (0 = full source)")
		outFile  = flag.String("out", "", "output JSON file (default: auto-named)")
		workers  = flag.Int("workers", 1, "parallel whisper workers (keep at 1 for CUDA)")
		threads  = flag.Int("threads", 8, "CPU threads per whisper worker")
		source   = flag.String(
			"source",
			"ytdlp",
			"media source type: ytdlp | podcast | local_file | direct_url",
		)

		// Whisper decoding
		language      = flag.String("language", "", "whisper language hint (e.g. en, es, auto); empty = auto-detect")
		translate     = flag.Bool("translate", false, "translate non-English audio into English")
		beamSize      = flag.Int("beam", 0, "beam search width (0 or 1 = greedy)")
		temperature   = flag.Float64("temperature", 0, "whisper sampling temperature (0 = greedy)")
		initialPrompt = flag.String("prompt", "", "initial prompt seeded to whisper")

		// VAD — presence of -vad-model enables VAD; the tuners default to library defaults when 0.
		vadModel             = flag.String("vad-model", "", "silero VAD model path (empty = VAD off)")
		vadThreshold         = flag.Float64("vad-threshold", 0, "VAD speech-probability cutoff (0..1, 0 = library default)")
		vadMinSpeechMs       = flag.Int("vad-min-speech-ms", 0, "minimum speech segment length (ms)")
		vadMinSilenceMs      = flag.Int("vad-min-silence-ms", 0, "minimum silence to end a speech region (ms)")
		vadMaxSpeechSec      = flag.Float64("vad-max-speech-sec", 0, "max single speech segment length (s)")
		vadSpeechPadMs       = flag.Int("vad-speech-pad-ms", 0, "padding around detected speech (ms)")
		vadSamplesOverlapSec = flag.Float64("vad-samples-overlap-sec", 0, "inter-segment overlap (s)")

		showProgress = flag.Bool("progress", false, "print progress events to stderr")
	)
	flag.Parse()

	if *mediaURL == "" {
		fmt.Fprintln(os.Stderr, "error: -url is required")
		flag.Usage()
		os.Exit(1)
	}

	var sourceOption streamscribe.Option
	switch *source {
	case "ytdlp":
		sourceOption = streamscribe.WithYtDlp()
	case "podcast":
		sourceOption = streamscribe.WithPodcastRSS()
	case "local_file":
		sourceOption = streamscribe.WithLocalFile()
	case "direct_url":
		sourceOption = streamscribe.WithDirectURL()
	default:
		flag.Usage()
		os.Exit(1)
	}

	s, err := streamscribe.New(
		*modelPath,
		streamscribe.WithWorkers(*workers),
		streamscribe.WithThreads(*threads),
		streamscribe.WithLogger(slog.Default()),
		sourceOption,
	)
	if err != nil {
		slog.Default().Error("failed to load model", "err", err)
		os.Exit(1)
	}
	defer s.Close()

	start := time.Duration(*startSec * float64(time.Second))
	var end time.Duration
	if *endSec > 0 {
		end = time.Duration(*endSec * float64(time.Second))
		if end <= start {
			fmt.Fprintln(os.Stderr, "error: -end must be greater than -start")
			os.Exit(1)
		}
	}

	if *outFile == "" {
		safeURL := *mediaURL
		for _, r := range []rune{'/', ':', '?', '&', '=', '%'} {
			safeURL = strings.ReplaceAll(safeURL, string(r), "_")
		}
		*outFile = fmt.Sprintf(
			"transcription_%s_%.0f-%.0f.json",
			safeURL,
			start.Seconds(),
			end.Seconds(),
		)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	transcribeOpts := []streamscribe.TranscribeOption{streamscribe.WithStart(start)}
	if end > 0 {
		transcribeOpts = append(transcribeOpts, streamscribe.WithEnd(end))
	}
	if *language != "" {
		transcribeOpts = append(transcribeOpts, streamscribe.WithLanguage(*language))
	}
	if *translate {
		transcribeOpts = append(transcribeOpts, streamscribe.WithTranslate(true))
	}
	if *beamSize > 0 {
		transcribeOpts = append(transcribeOpts, streamscribe.WithBeamSize(*beamSize))
	}
	if *temperature > 0 {
		transcribeOpts = append(transcribeOpts, streamscribe.WithTemperature(float32(*temperature)))
	}
	if *initialPrompt != "" {
		transcribeOpts = append(transcribeOpts, streamscribe.WithInitialPrompt(*initialPrompt))
	}
	if *vadModel != "" {
		var vadTuners []streamscribe.VADOption
		if *vadThreshold > 0 {
			vadTuners = append(vadTuners, streamscribe.VADThreshold(float32(*vadThreshold)))
		}
		if *vadMinSpeechMs > 0 {
			vadTuners = append(vadTuners, streamscribe.VADMinSpeechMs(*vadMinSpeechMs))
		}
		if *vadMinSilenceMs > 0 {
			vadTuners = append(vadTuners, streamscribe.VADMinSilenceMs(*vadMinSilenceMs))
		}
		if *vadMaxSpeechSec > 0 {
			vadTuners = append(vadTuners, streamscribe.VADMaxSpeechSec(float32(*vadMaxSpeechSec)))
		}
		if *vadSpeechPadMs > 0 {
			vadTuners = append(vadTuners, streamscribe.VADSpeechPadMs(*vadSpeechPadMs))
		}
		if *vadSamplesOverlapSec > 0 {
			vadTuners = append(vadTuners, streamscribe.VADSamplesOverlapSec(float32(*vadSamplesOverlapSec)))
		}
		transcribeOpts = append(transcribeOpts, streamscribe.WithVAD(*vadModel, vadTuners...))
	}
	if *showProgress {
		progressCh := make(chan streamscribe.Progress, 64)
		go func() {
			for p := range progressCh {
				slog.Info("progress",
					"stage", p.Stage,
					"chunk_idx", p.ChunkIdx,
					"chunk_total", p.ChunkTotal,
					"elapsed", p.Elapsed.Round(time.Millisecond),
				)
			}
		}()
		transcribeOpts = append(transcribeOpts, streamscribe.WithProgress(progressCh))
	}

	if err := s.TranscribeToFile(ctx, *mediaURL, *outFile, transcribeOpts...); err != nil {
		slog.Error("transcription failed", "err", err)
		os.Exit(1)
	}
}
