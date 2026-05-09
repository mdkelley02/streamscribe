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

	if err := s.TranscribeToFile(ctx, *mediaURL, *outFile, transcribeOpts...); err != nil {
		slog.Error("transcription failed", "err", err)
		os.Exit(1)
	}
}
