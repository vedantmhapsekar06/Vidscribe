# VidScribe AI

Turn long YouTube videos into structured learning notes: an AI summary plus timestamped highlights that link back to the exact moment in the video.

## How it works
1. Paste a YouTube URL and hit **Analyse**.
2. The FastAPI backend validates the URL and fetches the video's captions (`youtube-transcript-api`).
3. The transcript is split into token-bounded chunks, keeping each segment's start time.
4. **Map:** each chunk is summarised with `facebook/bart-large-cnn`, either through the Hugging Face Inference API or locally (see *Summarizer modes*).
5. **Reduce:** chunk summaries are merged and re-summarised (hierarchically if needed) into one summary.
6. **Highlights:** each summary sentence is aligned to the transcript window that best matches it, and the window's real start time becomes the timestamp. BART writes the sentence; it does not verify anything about the video.

## Tech stack
Python · FastAPI · Uvicorn · youtube-transcript-api · BART Large CNN · Hugging Face Inference API (or local Transformers + PyTorch) · python-dotenv · HTML/CSS/vanilla JS

## Setup
```bash
git clone https://github.com/<your-username>/VidScribe-AI.git
cd VidScribe-AI
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
python -m scripts.smoke_test  # verifies the selected summarizer works
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000. Interactive API docs are at `/docs`.

## Summarizer modes
Set `SUMMARIZER_BACKEND` in `.env`:

| Mode | What it does | Needs |
|---|---|---|
| `local` (default in `.env.example`) | Runs `facebook/bart-large-cnn` on your machine with PyTorch + Transformers | No token or payment. One-time ~1.6 GB model download, a few GB of RAM. Slower on CPU, much faster on an NVIDIA GPU. |
| `hf_api` | Calls the Hugging Face Inference API | `HF_API_TOKEN` from https://huggingface.co/settings/tokens **and** Inference Providers credits. Free accounts get little or no allowance, so a 402 error means you need to add credits at https://huggingface.co/billing. |

Both modes use the same model, chunking, map-reduce and highlight logic. The summary card shows which provider actually ran.

## Configuration
Settings go in `.env` (`SUMMARIZER_BACKEND`, and `HF_API_TOKEN` for API mode). Tunables (chunk size, summary lengths, retries, timeouts, highlight count) live in `backend/config.py`.

## API
`POST /process-video` with `{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}`

Success:
```json
{
  "success": true,
  "video_id": "VIDEO_ID",
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "summary": "...",
  "highlights": [{"timestamp": "00:02:14", "seconds": 134, "text": "...", "context": "..."}],
  "chunks_processed": 4,
  "model": "facebook/bart-large-cnn",
  "provider": "Hugging Face Inference API"
}
```
Errors return `{"success": false, "error": {"code": "...", "message": "..."}}`:

| Status | Meaning |
|---|---|
| 400 | Invalid URL or request body |
| 404 | Video private, deleted or unavailable |
| 422 | No captions / captions disabled / no English / too short / too long |
| 500 | Server configuration problem (e.g. missing or rejected token) |
| 502 | Hugging Face or YouTube upstream failure |
| 503 | Rate limited, model loading, quota used up, or YouTube blocking |
| 504 | Summarization service timed out |

## Testing
```bash
pip install -r requirements-dev.txt
python -m pytest
```
The tests mock YouTube and the summarizers, so they need no network, token or model download.

## Troubleshooting
- **`CONFIG_ERROR`**: in `hf_api` mode `HF_API_TOKEN` is empty, or `SUMMARIZER_BACKEND` has an unknown value. Restart Uvicorn after editing `.env`.
- **`LOCAL_MODEL_UNAVAILABLE`**: PyTorch/Transformers aren't installed in the active environment. Re-run `pip install -r requirements.txt`.
- **`LOCAL_MODEL_LOAD_FAILED`**: the first local run couldn't download the model. Check your internet connection and free disk space (about 2 GB).
- **Local mode is slow**: CPU inference takes several seconds per chunk, and the first request also loads the model. Long videos can take many minutes.
- **`HF_AUTH_ERROR`**: the token is invalid or lacks inference permission.
- **`HF_QUOTA` (HTTP 402)**: your Hugging Face account has no Inference Providers credits. Add credits at https://huggingface.co/billing, or switch to `SUMMARIZER_BACKEND=local`.
- **`MODEL_UNAVAILABLE`**: the hosted model isn't served to your account right now. Run `python -m scripts.smoke_test` to see.
- **`MODEL_LOADING` / `RATE_LIMITED`**: transient; the server retries with backoff, then asks you to try again.
- **`YOUTUBE_BLOCKED`**: YouTube sometimes blocks transcript requests from cloud/VPN IPs. Running locally on a home connection normally works.
- **`NO_TRANSCRIPT` / `CAPTIONS_DISABLED`**: the video has no captions; nothing is fabricated.
- **PowerShell says scripts are disabled** when activating the venv: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (`.venv\Scripts\activate.bat`).

## Limitations
- Quality depends on the video having captions; auto-generated captions are noisier.
- Local mode is limited by your hardware; API mode is limited by your credits.
- BART is English-only; non-English videos are used only if YouTube can translate their captions to English.
- Very long videos (over ~60 chunks) are rejected to keep requests reasonable.
