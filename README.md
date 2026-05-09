# streamscribe

A Go package for transcribing audio from streaming media platforms using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with CUDA acceleration.

Point it at a URL from any supported platform and receive an ordered stream of timestamped transcript segments — even for multi-hour content.

## Features

- **Multi-platform** — any of yt-dlp's 1000+ supported sites, plus podcast RSS feeds, local files, and raw media URLs
- **Arbitrarily long content** — streaming 30-second overlap pipeline means no memory ceiling
- **CUDA-accelerated** — runs on NVIDIA GPU via whisper.cpp's CUDA backend
- **Ordered output** — segments are always emitted in ascending timestamp order
- **Cancellable** — all goroutines respect `context.Context`; safe to interrupt mid-stream

## Supported Platforms

| Source | Tool | Constructor |
| ------ | ---- | ----------- |
| Any yt-dlp site (YouTube, Twitch, Vimeo, SoundCloud, Kick, X Spaces, [1000+ more](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)) | `yt-dlp` | `extractor.NewYtDlp()` |
| Podcast RSS feed | *(none)* | `extractor.NewPodcastRSS()` |
| Local file | *(none)* | `extractor.NewLocalFile()` |
| Raw media URL (HTTP, HLS, DASH, …) | *(none)* | `extractor.NewDirectURL()` |

## Prerequisites

### System dependencies

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** — [WSL2 install instructions](#cuda-on-wsl2)
- **ffmpeg** — audio decoding and resampling
- **whisper.cpp model file** — download from the [whisper.cpp models page](https://huggingface.co/ggerganov/whisper.cpp)

### Platform resolver tools

Install only the tools for platforms you intend to use:

```sh
pip install yt-dlp
```

### Recommended model

`ggml-large-v3-turbo.bin` fits in 12 GB VRAM and gives the best accuracy-to-speed ratio. `ggml-tiny.en.bin` is significantly faster and sufficient for English-only content.

## Usage

```go
package main

import (
    "context"
    "fmt"
    "os/signal"
    "syscall"
    "time"

    "github.com/mdkelley02/streamscribe"
)

func main() {
    ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
    defer cancel()

    s, err := streamscribe.New("models/ggml-large-v3-turbo.bin")
    if err != nil {
        panic(err)
    }
    defer s.Close()

    // Transcribe the first 30 minutes of a YouTube video.
    // WithStart / WithEnd are optional — omit both to transcribe the full source.
    segments, errCh := s.Transcribe(ctx, "https://www.youtube.com/watch?v=...",
        streamscribe.WithStart(0),
        streamscribe.WithEnd(30*time.Minute),
    )

    for seg := range segments {
        fmt.Printf("[%s --> %s] %s\n", seg.Start, seg.End, seg.Text)
    }

    if err := <-errCh; err != nil {
        panic(err)
    }
}
```

### Saving to a file

```go
err := s.TranscribeToFile(ctx, "https://www.youtube.com/watch?v=...", "output.json")
```

### Other media sources

```go
// Podcast RSS feed — transcribes the most recent episode
s, _ := streamscribe.New("models/ggml-large-v3-turbo.bin", streamscribe.WithPodcastRSS())
segments, errCh := s.Transcribe(ctx, "https://feeds.example.com/podcast.rss")

// Local file
s, _ := streamscribe.New("models/ggml-large-v3-turbo.bin", streamscribe.WithLocalFile())
segments, errCh := s.Transcribe(ctx, "/path/to/recording.mp3")

// Any URL ffmpeg can open directly (HLS, DASH, MP4, …)
s, _ := streamscribe.New("models/ggml-large-v3-turbo.bin", streamscribe.WithDirectURL())
segments, errCh := s.Transcribe(ctx, "https://cdn.example.com/audio.mp4")

// Custom resolver — implement your own platform support
s, _ := streamscribe.New("models/ggml-large-v3-turbo.bin",
    streamscribe.WithCustomSource(func(ctx context.Context, url string) (string, error) {
        return myPlatform.ResolveAudioURL(ctx, url)
    }),
)
```

## Adding a New Platform

Pass a resolver function to `WithCustomSource`. The resolver receives the URL and must return a direct media URL that ffmpeg can open (HLS manifest, MP4, RTMP, etc.).

```go
s, err := streamscribe.New("models/ggml-large-v3-turbo.bin",
    streamscribe.WithCustomSource(func(ctx context.Context, url string) (string, error) {
        // Call whatever CLI tool or API resolves this platform's URLs.
        out, err := exec.CommandContext(ctx, "some-tool", "--get-url", url).Output()
        if err != nil {
            return "", err
        }
        return strings.TrimSpace(string(out)), nil
    }),
)
```

## Architecture

```
URL
 │
 ▼
resolveURL()          platform-specific (streamlink, yt-dlp, ...)
 │
 ▼
ffmpeg (stdout pipe)  16 kHz mono f32le PCM
 │
 ▼
ExtractChunks         30 s windows, 28 s step, 2 s overlap
 │  chunkCh
 ▼
TranscribeChunks      whisper.cpp worker (CUDA), ordered merger
 │  segCh
 ▼
Segment{ Start, End, Text }
```

### Chunking design

whisper.cpp's encoder processes exactly 30 seconds of audio per call. A sliding window with a 2-second overlap between consecutive chunks ensures words that straddle a boundary are always seen in context. Each chunk is responsible for only its non-overlapping 28-second zone; the final chunk emits everything remaining.

## Building

### CUDA on WSL2

```sh
# Install CUDA toolkit (driver is already present via WSL)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda/bin:$PATH
```

### Build

```sh
# Builds whisper.cpp with CUDA then compiles the Go binary
make build
```

To target a different GPU architecture, edit `CMAKE_CUDA_ARCHITECTURES` in the Makefile:

| Value | GPUs                         |
| ----- | ---------------------------- |
| `86`  | RTX 30-series (Ampere)       |
| `89`  | RTX 40-series (Ada Lovelace) |
| `90`  | H100 (Hopper)                |

## License

MIT
