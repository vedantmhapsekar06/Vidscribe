"""Chunking, BART map-reduce summarisation, and transcript-grounded highlight extraction."""
import logging
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import InferenceClient

from .config import settings
from .errors import AppError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- tokens
@lru_cache(maxsize=1)
def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(settings.model_name)
    except Exception:
        logger.warning("BART tokenizer unavailable; using a word-based token estimate.")
        return None


def count_tokens(text: str) -> int:
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        return math.ceil(len(text.split()) * 1.4)
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    return len(ids)


def format_timestamp(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ------------------------------------------------------------------------ chunking
@dataclass
class Chunk:
    segments: list
    tokens: int

    @property
    def text(self) -> str:
        return " ".join(s["text"] for s in self.segments)


def _sentence_cut(segments, token_counts, max_tokens) -> int:
    """How many segments go in the finished chunk: prefer ending on a sentence boundary."""
    floor = 0.6 * max_tokens
    cumulative, total = [], 0
    for t in token_counts:
        total += t
        cumulative.append(total)
    for i in range(len(segments) - 1, -1, -1):
        if cumulative[i] < floor:
            break
        if segments[i]["text"].rstrip().endswith((".", "?", "!")):
            return i + 1
    return len(segments)


def chunk_transcript(segments, max_tokens, min_tail_tokens=0, counter=count_tokens):
    """Group whole segments into chunks of <= max_tokens, keeping each segment's timestamp."""
    chunks: list[Chunk] = []
    cur: list[dict] = []
    cur_tok: list[int] = []

    for seg in segments:
        t = counter(seg["text"])
        if cur and sum(cur_tok) + t > max_tokens:
            cut = _sentence_cut(cur, cur_tok, max_tokens)
            chunks.append(Chunk(cur[:cut], sum(cur_tok[:cut])))
            cur, cur_tok = cur[cut:], cur_tok[cut:]
        cur.append(seg)
        cur_tok.append(t)
    if cur:
        chunks.append(Chunk(cur, sum(cur_tok)))

    # Fold a tiny trailing chunk into its predecessor if it still fits comfortably.
    if (
        len(chunks) > 1
        and chunks[-1].tokens < min_tail_tokens
        and chunks[-2].tokens + chunks[-1].tokens <= max_tokens * 1.15
    ):
        tail = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(prev.segments + tail.segments, prev.tokens + tail.tokens)
    return chunks


# --------------------------------------------------------------- Hugging Face client
def _require_token() -> None:
    if not settings.hf_api_token:
        raise AppError(
            "CONFIG_ERROR",
            "The server is missing its Hugging Face API token. Set HF_API_TOKEN in the .env file.",
            500,
        )


def ensure_configured() -> None:
    """Fail fast on bad configuration, before any network or model work."""
    backend = settings.summarizer_backend
    if backend == "hf_api":
        _require_token()
    elif backend != "local":
        raise AppError(
            "CONFIG_ERROR",
            "Unknown SUMMARIZER_BACKEND setting. Use 'hf_api' or 'local' in the .env file.",
            500,
        )


def _status_code(exc):
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _is_timeout(exc) -> bool:
    return any("timeout" in cls.__name__.lower() for cls in type(exc).__mro__)


def _classify(exc):
    """Map a client exception to (user-safe AppError, retryable?)."""
    status = _status_code(exc)
    if _is_timeout(exc):
        return AppError("HF_TIMEOUT", "The summarization service took too long to respond. Please try again.", 504), True
    if status in (401, 403):
        return AppError("HF_AUTH_ERROR", "The summarization service rejected the server's credentials. Check the HF_API_TOKEN setting.", 500), False
    if status == 402:
        return AppError("HF_QUOTA", "The Hugging Face usage quota for this token has been used up.", 503), False
    if status == 404:
        return AppError("MODEL_UNAVAILABLE", "The summarization model isn't available from the configured provider.", 502), False
    if status == 429:
        return AppError("RATE_LIMITED", "The summarization service is rate limiting requests. Please wait a moment and try again.", 503), True
    if status == 503:
        return AppError("MODEL_LOADING", "The summarization model is still loading or temporarily unavailable. Please try again shortly.", 503), True
    if status in (500, 502, 504):
        return AppError("UPSTREAM_ERROR", "The summarization service had a temporary problem. Please try again.", 502), True
    if status is not None and 400 <= status < 500:
        return AppError("SUMMARIZATION_FAILED", "The summarization service couldn't process this request.", 502), False
    return AppError("UPSTREAM_ERROR", "Couldn't reach the summarization service. Please try again.", 502), True


def _lengths(tokens: int, cap_max: int, cap_min: int):
    max_len = max(20, min(cap_max, int(tokens * 0.6)))
    min_len = max(10, min(cap_min, max_len // 2))
    return max_len, min_len


def _extract_text(output) -> str:
    text = output.get("summary_text") if isinstance(output, dict) else getattr(output, "summary_text", None)
    return (text or "").strip()


class HFApiSummarizer:
    """Summarises through the Hugging Face Inference API (needs a token and credits)."""

    provider_label = settings.provider_label
    workers = settings.max_workers

    def __init__(self):
        _require_token()
        self._client = InferenceClient(
            provider=settings.hf_provider,
            token=settings.hf_api_token,
            timeout=settings.request_timeout,
        )

    def summarize(self, text: str, tokens: int, cap_max: int, cap_min: int) -> str:
        max_len, min_len = _lengths(tokens, cap_max, cap_min)
        timeouts = 0
        for attempt in range(settings.max_retries + 1):
            try:
                output = self._client.summarization(
                    text,
                    model=settings.model_name,
                    generate_parameters={"max_length": max_len, "min_length": min_len},
                )
            except Exception as exc:
                error, retryable = _classify(exc)
                if _is_timeout(exc):
                    timeouts += 1
                    retryable = timeouts < 2  # a 60 s timeout shouldn't be retried many times
                if retryable and attempt < settings.max_retries:
                    delay = min(settings.backoff_cap, settings.backoff_base * (2 ** attempt))
                    logger.warning("HF call failed (%s); retry %d in %.0fs", type(exc).__name__, attempt + 1, delay)
                    time.sleep(delay)
                    continue
                logger.error("HF call failed permanently", exc_info=exc)
                raise error from exc

            summary = _extract_text(output)
            if not summary:
                raise AppError("EMPTY_RESPONSE", "The summarization service returned an empty response.", 502)
            return summary
        raise AppError("UPSTREAM_ERROR", "The summarization service is unavailable. Please try again.", 502)


# ------------------------------------------------------------- local (no token) mode
class LocalSummarizer:
    """Runs facebook/bart-large-cnn on this machine with PyTorch + Transformers (free, no token)."""

    provider_label = settings.local_provider_label
    workers = 1  # one model instance; threads wouldn't speed up CPU inference

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise AppError(
                    "LOCAL_MODEL_UNAVAILABLE",
                    "Local mode needs PyTorch and Transformers. Run: pip install -r requirements.txt",
                    500,
                ) from exc
            try:
                logger.info("Loading %s locally (the first run downloads ~1.6 GB)...", settings.model_name)
                tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
            except Exception as exc:
                logger.error("Local model load failed", exc_info=exc)
                raise AppError(
                    "LOCAL_MODEL_LOAD_FAILED",
                    "Couldn't load the BART model locally. The first run needs an internet connection and "
                    "about 2 GB of free disk space to download it.",
                    500,
                ) from exc
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device).eval()
            self._tokenizer = tokenizer

    def summarize(self, text: str, tokens: int, cap_max: int, cap_min: int) -> str:
        self._ensure_loaded()
        import torch

        max_len, min_len = _lengths(tokens, cap_max, cap_min)
        try:
            with self._run_lock, torch.no_grad():
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
                inputs = inputs.to(self._device)
                output_ids = self._model.generate(
                    **inputs,
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=4,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            summary = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except AppError:
            raise
        except Exception as exc:
            logger.error("Local summarization failed", exc_info=exc)
            raise AppError("SUMMARIZATION_FAILED", "The local model couldn't summarise this text.", 500) from exc
        if not summary:
            raise AppError("EMPTY_RESPONSE", "The local model returned an empty summary.", 500)
        return summary


_local_summarizer = None
_local_lock = threading.Lock()


def get_summarizer():
    """Pick the summarizer from SUMMARIZER_BACKEND. The local model is kept loaded between requests."""
    global _local_summarizer
    ensure_configured()
    if settings.summarizer_backend == "local":
        with _local_lock:
            if _local_summarizer is None:
                _local_summarizer = LocalSummarizer()
            return _local_summarizer
    return HFApiSummarizer()


# ------------------------------------------------------------------- map / reduce
def _parallel(fn, items, workers):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fn, item) for item in items]
        try:
            return [f.result() for f in futures]
        except Exception:
            for f in futures:
                f.cancel()
            raise


def summarize_chunks(summarizer, chunks):
    return _parallel(
        lambda c: summarizer.summarize(c.text, c.tokens, settings.chunk_summary_max, settings.chunk_summary_min),
        chunks,
        summarizer.workers,
    )


def _group_by_tokens(texts, max_tokens):
    groups, cur, cur_tokens = [], [], 0
    for text in texts:
        t = count_tokens(text)
        if cur and cur_tokens + t > max_tokens:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(text)
        cur_tokens += t
    if cur:
        groups.append(cur)
    return groups


def _reduce_level(summarizer, texts):
    def work(group):
        if len(group) == 1:
            return group[0]
        combined = " ".join(group)
        return summarizer.summarize(
            combined, count_tokens(combined), settings.chunk_summary_max, settings.chunk_summary_min
        )

    return _parallel(work, _group_by_tokens(texts, settings.max_chunk_tokens), summarizer.workers)


def reduce_summaries(summarizer, summaries):
    """Hierarchical reduce: shrink level by level until one final pass fits the model limit."""
    current = [s for s in summaries if s]
    for _ in range(settings.max_reduce_levels):
        if len(current) == 1:
            return current[0]
        combined = " ".join(current)
        total = count_tokens(combined)
        if total <= settings.max_chunk_tokens:
            return summarizer.summarize(combined, total, settings.final_summary_max, settings.final_summary_min)
        current = _reduce_level(summarizer, current)
    raise AppError("SUMMARIZATION_FAILED", "Could not condense this transcript into a single summary.", 502)


# ------------------------------------------------------------------- highlights
_STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not now of off on once only or other our out
    over own same she should so some such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who why will with would you your yeah okay like um uh
    going gonna get got one really know think""".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Window:
    chunk_index: int
    start: float
    text: str
    terms: set


def _terms(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _build_windows(chunks):
    windows = []
    size, stride = settings.window_segments, settings.window_stride
    for ci, chunk in enumerate(chunks):
        segs = chunk.segments
        for i in range(0, len(segs), stride):
            part = segs[i : i + size]
            text = " ".join(s["text"] for s in part)
            windows.append(_Window(ci, part[0]["start"], text, _terms(text)))
            if i + size >= len(segs):
                break
    return windows


def _idf(windows):
    df = Counter()
    for w in windows:
        df.update(w.terms)
    n = len(windows)
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def _finalise(candidates):
    candidates.sort(key=lambda c: c["start"])
    spaced = []
    for cand in candidates:
        if spaced and cand["start"] - spaced[-1]["start"] < settings.min_highlight_gap_s:
            if cand["score"] > spaced[-1]["score"]:
                spaced[-1] = cand
            continue
        spaced.append(cand)
    if len(spaced) > settings.max_highlights:
        top = sorted(spaced, key=lambda c: c["score"], reverse=True)[: settings.max_highlights]
        spaced = sorted(top, key=lambda c: c["start"])
    return [
        {
            "timestamp": format_timestamp(c["start"]),
            "seconds": int(c["start"]),
            "text": c["text"],
            "context": c["context"],
        }
        for c in spaced
    ]


def extract_highlights(chunks, chunk_summaries):
    """Align each summary sentence to the transcript window that best matches it.

    The timestamp is the real start time of that window, so every highlight is
    traceable to actual transcript text; BART is only used to write the sentence.
    """
    windows = _build_windows(chunks)
    idf = _idf(windows)
    default_idf = math.log(len(windows) + 1) + 1
    by_chunk: dict[int, list] = {}
    for w in windows:
        by_chunk.setdefault(w.chunk_index, []).append(w)

    quota = max(1, math.ceil(settings.max_highlights / len(chunks)))
    candidates = []
    for ci, summary in enumerate(chunk_summaries):
        best_by_window = {}
        for sentence in _SENTENCE_RE.split(summary):
            sentence = sentence.strip()
            terms = _terms(sentence)
            if not terms:
                continue
            denom = sum(idf.get(t, default_idf) for t in terms)
            window, score = max(
                ((w, sum(idf[t] for t in terms & w.terms) / denom) for w in by_chunk[ci]),
                key=lambda pair: pair[1],
            )
            current = best_by_window.get(window.start)
            if current is None or score > current["score"]:
                best_by_window[window.start] = {
                    "score": score,
                    "start": window.start,
                    "text": sentence,
                    "context": _clip(window.text, settings.context_chars),
                }
        ranked = sorted(best_by_window.values(), key=lambda c: c["score"], reverse=True)
        for rank, cand in enumerate(ranked[:quota]):
            if rank == 0 or cand["score"] >= settings.min_extra_score:
                candidates.append(cand)
    return _finalise(candidates)


# ------------------------------------------------------------------- entry point
def generate_notes(segments) -> dict:
    ensure_configured()
    chunks = chunk_transcript(segments, settings.max_chunk_tokens, settings.min_tail_tokens)

    if sum(c.tokens for c in chunks) < settings.min_transcript_tokens:
        raise AppError("TRANSCRIPT_TOO_SHORT", "This video's transcript is too short to summarise.", 422)
    if len(chunks) > settings.max_chunks:
        raise AppError("VIDEO_TOO_LONG", "This video is too long to process. Try a shorter one.", 422)

    summarizer = get_summarizer()
    chunk_summaries = summarize_chunks(summarizer, chunks)
    summary = reduce_summaries(summarizer, chunk_summaries)
    highlights = extract_highlights(chunks, chunk_summaries)
    return {
        "summary": summary,
        "highlights": highlights,
        "chunks_processed": len(chunks),
        "provider": summarizer.provider_label,
    }
