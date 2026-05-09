package streamscribe

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func Test_Transcribe(t *testing.T) {
	for _, tc := range []struct {
		ModelPath string
		Options   []Option
		URL       string
		Want      []Segment
	}{
		{
			ModelPath: "whisper.cpp/models/ggml-tiny.en.bin",
			Options:   []Option{WithYtDlp()},
			URL:       "https://www.youtube.com/watch?v=jNQXAC9IVRw",
			Want: []Segment{
				{
					Start: 0,
					End:   3640000000,
					Text:  "Alright so here we are one of the elephants.",
				},
				{
					Start: 3640000000,
					End:   17160000000,
					Text:  "The cool thing about these guys is that they have really really long pumps and that's cool.",
				},
				{
					Start: 17160000000,
					End:   19120000000,
					Text:  "And that's pretty much all there is to say.",
				},
			},
		},
		{
			ModelPath: "whisper.cpp/models/ggml-large-v3-turbo.bin",
			Options:   []Option{WithYtDlp()},
			URL:       "https://www.youtube.com/watch?v=jNQXAC9IVRw",
			Want: []Segment{
				{
					Start: 1000000000,
					End:   4000000000,
					Text:  "Alright, so here we are, one of the elephants.",
				},
				{
					Start: 4000000000,
					End:   12000000000,
					Text:  "The cool thing about these guys is that they have really, really, really long prongs."},
				{
					Start: 12000000000,
					End:   15000000000,
					Text:  "And that's cool.",
				},
				{
					Start: 15000000000,
					End:   20000000000,
					Text:  "And that's pretty much all there is to say.",
				},
			},
		},
	} {
		t.Run(tc.ModelPath+tc.URL, func(t *testing.T) {
			t.Parallel()

			s, err := New(tc.ModelPath, tc.Options...)
			require.NoError(t, err)
			t.Cleanup(func() {
				require.NoError(t, s.Close())
			})

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			got, err := Drain(s.Transcribe(ctx, tc.URL))
			require.NoError(t, err)
			require.Equal(t, tc.Want, got)
		})
	}
}
