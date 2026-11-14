"""Central configuration. Tunables live here so other modules stay free of magic numbers."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # "hf_api" (Hugging Face Inference API, needs a token + credits) or "local" (runs BART on this machine)
    summarizer_backend: str = (os.getenv("SUMMARIZER_BACKEND", "").strip().lower() or "hf_api")

    # Credentials (server-side only, never sent to the browser)
    hf_api_token: str = os.getenv("HF_API_TOKEN", "").strip()

    # Model / provider
    model_name: str = "facebook/bart-large-cnn"
    hf_provider: str = "hf-inference"
    provider_label: str = "Hugging Face Inference API"
    local_provider_label: str = "Local (Transformers)"

    # Chunking (BART accepts at most 1024 tokens; keep headroom)
    max_chunk_tokens: int = 800
    min_tail_tokens: int = 120
    min_transcript_tokens: int = 50
    max_chunks: int = 60

    # Summary lengths (tokens)
    chunk_summary_max: int = 150
    chunk_summary_min: int = 40
    final_summary_max: int = 250
    final_summary_min: int = 60
    max_reduce_levels: int = 6

    # Reliability
    max_workers: int = 3
    request_timeout: float = 60.0
    max_retries: int = 4
    backoff_base: float = 2.0
    backoff_cap: float = 20.0

    # Highlights
    max_highlights: int = 10
    window_segments: int = 6
    window_stride: int = 3
    min_highlight_gap_s: int = 20
    min_extra_score: float = 0.25
    context_chars: int = 220


settings = Settings()
