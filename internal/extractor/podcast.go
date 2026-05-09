package extractor

import (
	"context"
	"encoding/xml"
	"fmt"
	"net/http"
	"strings"
)

// resolvePodcastRSSURL fetches an RSS feed and returns the audio URL of the
// most recent episode. It checks both <enclosure> (standard RSS) and
// <media:content> (Media RSS extension) to cover all common feed formats.

// ── RSS XML types ─────────────────────────────────────────────────────────────

type rssFeed struct {
	Channel rssChannel `xml:"channel"`
}

type rssChannel struct {
	Items []rssItem `xml:"item"`
}

type rssItem struct {
	// Standard RSS 2.0
	Enclosure rssEnclosure `xml:"enclosure"`
	// Media RSS extension (http://search.yahoo.com/mrss/)
	MediaContent rssMediaContent `xml:"http://search.yahoo.com/mrss/ content"`
}

// audioURL returns the first audio URL found in the item, checking both
// the standard <enclosure> and the Media RSS <media:content> element.
func (i rssItem) audioURL() string {
	if strings.HasPrefix(i.Enclosure.Type, "audio/") && i.Enclosure.URL != "" {
		return i.Enclosure.URL
	}
	if (i.MediaContent.Medium == "audio" || strings.HasPrefix(i.MediaContent.Type, "audio/")) &&
		i.MediaContent.URL != "" {
		return i.MediaContent.URL
	}
	return ""
}

type rssEnclosure struct {
	URL  string `xml:"url,attr"`
	Type string `xml:"type,attr"`
}

type rssMediaContent struct {
	URL    string `xml:"url,attr"`
	Medium string `xml:"medium,attr"`
	Type   string `xml:"type,attr"`
}

// NewPodcastRSS returns an extractor for podcast episodes via RSS feed URL.
// The URL passed to ExtractChunks must be the RSS feed URL; the resolver
// fetches the feed and returns the most recent episode's audio URL.
// No external tools required.
func NewPodcastRSS() ChunkExtractor {
	return &ffmpegExtractor{
		resolveURL: func(ctx context.Context, feedURL string) (string, error) {
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, feedURL, nil)
			if err != nil {
				return "", fmt.Errorf("build request: %w", err)
			}

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return "", fmt.Errorf("fetch feed: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return "", fmt.Errorf("feed returned HTTP %d", resp.StatusCode)
			}

			var feed rssFeed
			if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
				return "", fmt.Errorf("parse feed: %w", err)
			}

			for _, item := range feed.Channel.Items {
				if url := item.audioURL(); url != "" {
					return url, nil
				}
			}
			return "", fmt.Errorf("no audio enclosure found in RSS feed")
		},
	}
}
