import pytest

from backend.errors import AppError
from backend.note_generator import chunk_transcript, format_timestamp
from backend.youtube_handler import extract_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/shorts/{VID}",
    f"youtube.com/watch?v={VID}&t=30s",
])
def test_valid_urls(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize("url", [
    "", "not a url", f"https://example.com/watch?v={VID}",
    "https://www.youtube.com/watch?v=short", f"ftp://youtu.be/{VID}",
])
def test_invalid_urls(url):
    with pytest.raises(AppError) as exc:
        extract_video_id(url)
    assert exc.value.status_code == 400


def test_format_timestamp():
    assert format_timestamp(134) == "00:02:14"
    assert format_timestamp(3725.9) == "01:02:05"


def test_chunking_respects_limit_and_preserves_order():
    segments = [{"text": "word " * 10, "start": float(i * 5), "duration": 5.0} for i in range(100)]
    chunks = chunk_transcript(segments, max_tokens=100, counter=lambda s: len(s.split()))
    assert all(c.tokens <= 100 for c in chunks)
    starts = [s["start"] for c in chunks for s in c.segments]
    assert starts == [s["start"] for s in segments]
