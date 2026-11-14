"""End-to-end API test with YouTube and Hugging Face mocked out (no network, no token needed)."""
from fastapi.testclient import TestClient

from backend import app as app_module
from backend import note_generator
from backend.config import settings

VID = "dQw4w9WgXcQ"


def _segments():
    topics = ["photosynthesis chlorophyll sunlight", "calvin cycle glucose sugars", "mitochondria cellular respiration"]
    return [
        {"text": f"Now we discuss {topics[(i // 30) % 3]} in more depth and detail today.", "start": float(i * 4), "duration": 4.0}
        for i in range(120)
    ]


def test_process_video_success(monkeypatch):
    monkeypatch.setattr(note_generator, "ensure_configured", lambda: None)
    monkeypatch.setattr(app_module, "ensure_configured", lambda: None)
    monkeypatch.setattr(app_module, "get_transcript", lambda vid: _segments())

    class FakeSummarizer:
        provider_label = "Fake Provider"
        workers = 2

        def summarize(self, text, tokens, cap_max, cap_min):
            return "The speaker explains photosynthesis and chlorophyll. Sugars form in the calvin cycle."

    monkeypatch.setattr(note_generator, "get_summarizer", lambda: FakeSummarizer())

    client = TestClient(app_module.app)
    res = client.post("/process-video", json={"url": f"https://youtu.be/{VID}"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["success"] and data["video_id"] == VID
    assert data["summary"] and data["chunks_processed"] >= 1
    assert data["model"] == settings.model_name
    assert data["provider"] == "Fake Provider"
    assert data["highlights"]
    for h in data["highlights"]:
        assert h["timestamp"].count(":") == 2 and isinstance(h["seconds"], int)


def test_invalid_url_returns_400(monkeypatch):
    monkeypatch.setattr(app_module, "ensure_configured", lambda: None)
    client = TestClient(app_module.app)
    res = client.post("/process-video", json={"url": "https://example.com"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_URL"


def test_bad_body_returns_400():
    client = TestClient(app_module.app)
    assert client.post("/process-video", json={}).status_code == 400


def test_index_and_assets_served():
    client = TestClient(app_module.app)
    assert client.get("/").status_code == 200
    assert client.get("/static/vidscribe.js").status_code == 200
    assert client.get("/static/vidscribe.css").status_code == 200
