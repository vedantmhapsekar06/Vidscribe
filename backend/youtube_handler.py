"""YouTube URL validation and transcript retrieval (youtube-transcript-api 1.x)."""
import logging
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from .errors import AppError

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
_PATH_RE = re.compile(r"^/(?:embed|shorts)/([^/?#]+)")
_EN_CODES = ["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"]
_SOUND_TAG_RE = re.compile(r"\[[^\]]{1,30}\]")  # e.g. [Music], [Applause]

# Matched by class name so this works across youtube-transcript-api versions.
_ERROR_MAP = {
    "InvalidVideoId": ("INVALID_URL", "That doesn't look like a valid YouTube video URL.", 400),
    "VideoUnavailable": ("VIDEO_UNAVAILABLE", "This video is unavailable, private, or has been removed.", 404),
    "VideoUnplayable": ("VIDEO_UNAVAILABLE", "This video can't be played, so its transcript can't be retrieved.", 404),
    "AgeRestricted": ("AGE_RESTRICTED", "This video is age-restricted, so its transcript can't be retrieved.", 422),
    "TranscriptsDisabled": ("CAPTIONS_DISABLED", "Captions are disabled for this video, so no transcript is available.", 422),
    "NoTranscriptFound": ("NO_TRANSCRIPT", "No usable captions were found for this video.", 422),
    "NotTranslatable": ("NO_ENGLISH_TRANSCRIPT", "This video has no English captions and they can't be translated.", 422),
    "TranslationLanguageNotAvailable": ("NO_ENGLISH_TRANSCRIPT", "This video has no English captions and they can't be translated.", 422),
    "IpBlocked": ("YOUTUBE_BLOCKED", "YouTube is temporarily blocking transcript requests from this server. Try again later.", 503),
    "RequestBlocked": ("YOUTUBE_BLOCKED", "YouTube is temporarily blocking transcript requests from this server. Try again later.", 503),
    "TooManyRequests": ("YOUTUBE_BLOCKED", "YouTube is temporarily blocking transcript requests from this server. Try again later.", 503),
}


def extract_video_id(url: str) -> str:
    """Return the 11-character video ID for supported YouTube URL forms, else raise AppError(400)."""

    def invalid() -> AppError:
        return AppError(
            "INVALID_URL",
            "That doesn't look like a valid YouTube video URL. Use a youtube.com/watch, youtu.be, /embed or /shorts link.",
            400,
        )

    if not isinstance(url, str) or not url.strip() or len(url) > 300:
        raise invalid()

    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except ValueError:
        raise invalid()
    if parsed.scheme not in ("http", "https"):
        raise invalid()

    host = (parsed.hostname or "").lower()
    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        else:
            match = _PATH_RE.match(parsed.path)
            candidate = match.group(1) if match else ""

    if not _VIDEO_ID_RE.match(candidate):
        raise invalid()
    return candidate


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _select_transcript(transcript_list):
    """Prefer English (manual before auto-generated); otherwise translate to English."""
    try:
        return transcript_list.find_transcript(_EN_CODES)
    except Exception as exc:
        if type(exc).__name__ != "NoTranscriptFound":
            raise

    for transcript in sorted(transcript_list, key=lambda t: t.is_generated):
        if transcript.is_translatable and any(
            lang.language_code == "en" for lang in transcript.translation_languages
        ):
            return transcript.translate("en")

    raise AppError(
        "NO_ENGLISH_TRANSCRIPT",
        "This video has no English captions and none could be translated to English.",
        422,
    )


def _translate_error(exc: Exception) -> AppError:
    for cls in type(exc).__mro__:
        if cls.__name__ in _ERROR_MAP:
            code, message, status = _ERROR_MAP[cls.__name__]
            return AppError(code, message, status)
    logger.error("Unexpected transcript error", exc_info=exc)
    return AppError(
        "TRANSCRIPT_ERROR",
        "Could not retrieve the transcript from YouTube. Check your connection and try again.",
        502,
    )


def get_transcript(video_id: str) -> list[dict]:
    """Return [{'text', 'start', 'duration'}, ...] exactly as provided by YouTube captions."""
    try:
        api = YouTubeTranscriptApi()
        transcript = _select_transcript(api.list(video_id))
        raw = transcript.fetch().to_raw_data()
    except AppError:
        raise
    except Exception as exc:
        raise _translate_error(exc) from exc

    segments = []
    for item in raw:
        text = _SOUND_TAG_RE.sub(" ", str(item.get("text", "")))
        text = " ".join(text.split())
        if text:
            segments.append(
                {
                    "text": text,
                    "start": float(item.get("start", 0.0)),
                    "duration": float(item.get("duration", 0.0)),
                }
            )

    if not segments:
        raise AppError("EMPTY_TRANSCRIPT", "The transcript for this video is empty.", 422)
    return segments
