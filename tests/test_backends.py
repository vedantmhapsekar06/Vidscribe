"""Backend selection and local-mode behaviour (no network, no token, no model download)."""
from dataclasses import replace

import pytest

from backend import note_generator as ng
from backend.errors import AppError


def _use(monkeypatch, **overrides):
    monkeypatch.setattr(ng, "settings", replace(ng.settings, **overrides))
    monkeypatch.setattr(ng, "_local_summarizer", None)


def test_local_mode_needs_no_token(monkeypatch):
    _use(monkeypatch, summarizer_backend="local", hf_api_token="")
    ng.ensure_configured()  # must not raise
    summarizer = ng.get_summarizer()
    assert isinstance(summarizer, ng.LocalSummarizer)
    assert summarizer.workers == 1
    assert ng.get_summarizer() is summarizer  # model is loaded once and reused


def test_api_mode_requires_token(monkeypatch):
    _use(monkeypatch, summarizer_backend="hf_api", hf_api_token="")
    with pytest.raises(AppError) as exc:
        ng.get_summarizer()
    assert exc.value.code == "CONFIG_ERROR"


def test_api_mode_with_token_uses_api_summarizer(monkeypatch):
    _use(monkeypatch, summarizer_backend="hf_api", hf_api_token="hf_test")
    assert isinstance(ng.get_summarizer(), ng.HFApiSummarizer)


def test_unknown_backend_is_rejected(monkeypatch):
    _use(monkeypatch, summarizer_backend="banana")
    with pytest.raises(AppError) as exc:
        ng.ensure_configured()
    assert exc.value.code == "CONFIG_ERROR"
