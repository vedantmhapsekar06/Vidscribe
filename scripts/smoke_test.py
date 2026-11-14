"""Checks that your chosen summarizer works. Run: python -m scripts.smoke_test"""
import time

from backend.config import settings
from backend.note_generator import count_tokens, get_summarizer

SAMPLE = (
    "Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into "
    "chemical energy. In the chloroplasts, chlorophyll absorbs sunlight, mostly in the red and blue wavelengths. "
    "That energy drives the splitting of water molecules, releasing oxygen as a by-product. The resulting energy "
    "carriers, ATP and NADPH, power the Calvin cycle, where carbon dioxide from the air is fixed into sugars such "
    "as glucose. These sugars fuel the plant's growth and form the base of almost every food chain on Earth, "
    "which is why photosynthesis is considered one of the most important biological processes."
)


def main() -> None:
    print(f"Backend: {settings.summarizer_backend}   Model: {settings.model_name}")
    if settings.summarizer_backend == "local":
        print("Local mode: the first run downloads ~1.6 GB, so this can take a few minutes...")
    summarizer = get_summarizer()
    tokens = count_tokens(SAMPLE)
    print(f"Tokenizer OK ({tokens} tokens in sample)")
    started = time.time()
    summary = summarizer.summarize(SAMPLE, tokens, 80, 20)
    print(f"Provider: {summarizer.provider_label}   ({time.time() - started:.1f}s)")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
