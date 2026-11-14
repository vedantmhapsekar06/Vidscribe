(() => {
  "use strict";

  const CLIENT_TIMEOUT_MS = 20 * 60 * 1000;
  const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;
  const SUMMARY_PLACEHOLDER = "Your summary will appear here once the notes are generated.";

  const $ = (id) => document.getElementById(id);
  const sheet = $("sheet");
  const form = $("analyse-form");
  const urlInput = $("url-input");
  const analyseBtn = $("analyse-btn");
  const stateRow = $("state-row");
  const emptyMsg = $("empty-msg");
  const noMoments = $("no-moments");
  const statusBox = $("status");
  const statusText = $("status-text");
  const errorBox = $("error");
  const errorText = $("error-text");
  const retryBtn = $("retry-btn");
  const list = $("highlight-list");
  const highlightBadge = $("highlight-badge");
  const chunkBadge = $("chunk-badge");
  const summaryText = $("summary-text");
  const meta = $("meta");

  let lastUrl = "";
  let timerId = null;

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  // Which message the state row shows: "empty" | "none" | "loading" | "error" | null (hide the row).
  function showState(which) {
    stateRow.hidden = which === null;
    emptyMsg.hidden = which !== "empty";
    noMoments.hidden = which !== "none";
    statusBox.hidden = which !== "loading";
    errorBox.hidden = which !== "error";
  }

  function looksLikeYouTubeUrl(value) {
    try {
      const raw = /^[a-z]+:\/\//i.test(value) ? value : `https://${value}`;
      const host = new URL(raw).hostname.replace(/^www\./, "").toLowerCase();
      return ["youtube.com", "m.youtube.com", "youtu.be"].includes(host);
    } catch {
      return false;
    }
  }

  function startTimer() {
    const started = Date.now();
    const tick = () => {
      const s = Math.floor((Date.now() - started) / 1000);
      statusText.textContent = `Processing video… ${s}s elapsed. Longer videos may take several minutes.`;
    };
    tick();
    timerId = setInterval(tick, 1000);
  }

  function stopTimer() {
    clearInterval(timerId);
    timerId = null;
  }

  function setLoading(isLoading) {
    analyseBtn.disabled = isLoading;
    analyseBtn.textContent = isLoading ? "Processing…" : "Generate notes";
    form.setAttribute("aria-busy", String(isLoading));
    if (isLoading) {
      showState("loading");
      startTimer();
    } else {
      stopTimer();
    }
  }

  function showError(message, { retry = true } = {}) {
    errorText.textContent = message;
    retryBtn.hidden = !retry;
    showState("error");
  }

  // Clear any previous result so stale notes never sit next to a new request.
  function resetOutput() {
    list.replaceChildren();
    list.hidden = true;
    highlightBadge.hidden = true;
    chunkBadge.hidden = true;
    meta.hidden = true;
    summaryText.textContent = SUMMARY_PLACEHOLDER;
    summaryText.classList.add("placeholder");
  }

  function buildMoment(h, videoId) {
    const row = document.createElement("li");
    row.className = "row moment";

    const cue = document.createElement("div");
    cue.className = "cue";

    const seconds = Math.max(0, Math.floor(Number(h.seconds)));
    let stamp;
    if (videoId && Number.isFinite(seconds)) {
      stamp = document.createElement("a");
      stamp.href = `https://www.youtube.com/watch?v=${videoId}&t=${seconds}`;
      stamp.target = "_blank";
      stamp.rel = "noopener noreferrer";
      stamp.title = "Open the video at this moment";
    } else {
      stamp = document.createElement("span");
    }
    stamp.className = "timestamp";
    stamp.textContent = String(h.timestamp);
    cue.appendChild(stamp);

    const note = document.createElement("div");
    note.className = "note";
    const text = document.createElement("p");
    text.className = "moment-text";
    text.textContent = String(h.text);
    note.appendChild(text);

    if (h.context) {
      const ctx = document.createElement("p");
      ctx.className = "moment-context";
      ctx.textContent = `“${h.context}”`;
      note.appendChild(ctx);
    }

    row.append(cue, note);
    return row;
  }

  function renderResults(data) {
    summaryText.textContent = data.summary;
    summaryText.classList.remove("placeholder");
    chunkBadge.textContent = `${plural(data.chunks_processed, "transcript section")} analysed`;
    chunkBadge.hidden = false;
    $("meta-model").textContent = data.model;
    $("meta-provider").textContent = data.provider;
    meta.hidden = false;

    const videoId = VIDEO_ID_RE.test(data.video_id) ? data.video_id : null;
    const highlights = Array.isArray(data.highlights) ? data.highlights : [];
    highlightBadge.textContent = String(highlights.length);
    highlightBadge.hidden = false;

    if (highlights.length === 0) {
      showState("none");
    } else {
      highlights.forEach((h) => list.appendChild(buildMoment(h, videoId)));
      list.hidden = false;
      showState(null);
    }
    sheet.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function analyse(url) {
    lastUrl = url;
    resetOutput();
    setLoading(true);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);

    try {
      const res = await fetch("/process-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
        signal: controller.signal,
      });

      let data = null;
      try { data = await res.json(); } catch { /* non-JSON error body */ }

      if (!res.ok || !data || data.success !== true) {
        throw new Error(data?.error?.message || `The request failed (HTTP ${res.status}).`);
      }
      renderResults(data);
    } catch (err) {
      if (err.name === "AbortError") {
        showError("The request took too long. Please try again.");
      } else if (err instanceof TypeError) {
        showError("Unable to reach the server. Please confirm it is running and try again.");
      } else {
        showError(err.message || "Something went wrong. Please try again.");
      }
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = urlInput.value.trim();
    if (!value) {
      showError("Please enter a YouTube URL.", { retry: false });
      return;
    }
    if (!looksLikeYouTubeUrl(value)) {
      showError("Please enter a valid YouTube link (youtube.com or youtu.be).", { retry: false });
      return;
    }
    analyse(value);
  });

  retryBtn.addEventListener("click", () => {
    if (lastUrl) analyse(lastUrl);
  });
})();
