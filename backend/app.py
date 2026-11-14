"""FastAPI application: serves the frontend and exposes POST /process-video."""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .errors import AppError
from .note_generator import ensure_configured, generate_notes
from .youtube_handler import canonical_url, extract_video_id, get_transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vidscribe")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="VidScribe AI", version="1.0.0")

# The frontend is served from the same origin, so CORS is only needed if you open it from elsewhere locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class VideoRequest(BaseModel):
    url: str


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    return _error(400, "INVALID_REQUEST", 'Send JSON like {"url": "https://www.youtube.com/watch?v=VIDEO_ID"}.')


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception):
    logger.error("Unhandled error", exc_info=exc)
    return _error(500, "INTERNAL_ERROR", "Something went wrong on the server. Please try again.")


@app.post("/process-video")
def process_video(payload: VideoRequest):
    # Plain `def` so FastAPI runs this blocking work in its threadpool.
    ensure_configured()  # fail fast, before any network calls
    video_id = extract_video_id(payload.url)
    segments = get_transcript(video_id)
    notes = generate_notes(segments)
    return {
        "success": True,
        "video_id": video_id,
        "video_url": canonical_url(video_id),
        "summary": notes["summary"],
        "highlights": notes["highlights"],
        "chunks_processed": notes["chunks_processed"],
        "model": settings.model_name,
        "provider": notes["provider"],
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "dashboard.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
