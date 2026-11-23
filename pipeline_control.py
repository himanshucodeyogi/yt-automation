"""Shared cancellation support for the automation pipeline.

The web dashboard runs blocking pipeline work in worker threads.  A boolean
``stop`` flag alone is not enough because a long FFmpeg/MoviePy operation does
not return to the thread until it has finished.  ``CancellationToken`` combines
a thread-safe event with a registry of active child processes so a stop request
can interrupt work immediately.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Set


class PipelineCancelledError(Exception):
    """Raised when a user cancels a running pipeline operation."""


class CancellationToken:
    """Thread-safe cancellation token shared by the API and worker thread."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: Set[Any] = set()

    def cancel(self) -> None:
        """Request cancellation and terminate registered child processes."""
        self._event.set()

        with self._lock:
            processes = list(self._processes)

        for process in processes:
            self._terminate_process(process)

    def is_cancelled(self) -> bool:
        """Return ``True`` when cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise :class:`PipelineCancelledError` if cancellation was requested."""
        if self.is_cancelled():
            raise PipelineCancelledError("Pipeline operation was cancelled by the user")

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for cancellation and return its current state."""
        return self._event.wait(timeout)

    def register_process(self, process: Any) -> None:
        """Register a child process so it can be terminated on cancellation."""
        with self._lock:
            if not self.is_cancelled():
                self._processes.add(process)
                return

        # Cancellation may have happened between the check and registration.
        self._terminate_process(process)

    def unregister_process(self, process: Any) -> None:
        """Remove a completed child process from the cancellation registry."""
        with self._lock:
            self._processes.discard(process)

    @staticmethod
    def _terminate_process(process: Any) -> None:
        """Best-effort termination that also works for already-finished jobs."""
        try:
            if process.poll() is None:
                process.terminate()
        except (OSError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# Word-level caption timing helpers
#
# The animated captions are word-by-word, so every single word needs a
# timestamp.  Some transcription back-ends only return *segment* level
# timestamps.  These helpers turn a segment into word timings so captions
# always work, no matter which backend produced the transcript.
# ---------------------------------------------------------------------------

def split_text_for_captions(text: str) -> List[str]:
    """
    Split a segment's text into caption-sized tokens.

    Latin/Devanagari text splits on whitespace.  CJK text has no spaces, so
    it is split per character instead - otherwise the whole sentence would
    become one giant caption.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokens = text.split()

    # If there is no whitespace but the text contains CJK characters,
    # split it character by character.
    if len(tokens) == 1:
        cjk = any(
            "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u4dbf"
            or "\u4e00" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
            for ch in text
        )
        if cjk and len(text) > 1:
            return list(text)

    return tokens


def synthesize_word_timestamps(
    text: str,
    start: float,
    end: float,
    min_word_duration: float = 0.12,
) -> List[Dict[str, Any]]:
    """
    Distribute a segment's time range across its words.

    The segment duration is split between the words, weighted by word length
    (a rough proxy for how long a word takes to say).  Every word also gets
    a small guaranteed slice so that very short words stay visible.

    Args:
        text: The segment text
        start: Segment start time in seconds
        end: Segment end time in seconds
        min_word_duration: Minimum time a word should be visible for

    Returns:
        List of ``{"word", "start", "end"}`` dicts (empty if there is no text)
    """
    tokens = split_text_for_captions(text)
    if not tokens:
        return []

    start = float(start or 0.0)
    end = float(end if end is not None else start)
    duration = max(0.0, end - start)

    count = len(tokens)
    if duration <= 0:
        duration = min_word_duration * count
        end = start + duration

    # Every word gets a small guaranteed slice, the rest of the time is
    # distributed by word length.
    floor = min(min_word_duration, duration / count)
    remaining = max(0.0, duration - floor * count)

    weights = [max(1, len(token)) for token in tokens]
    total_weight = sum(weights)

    words: List[Dict[str, Any]] = []
    cursor = start
    for token, weight in zip(tokens, weights):
        word_duration = floor + remaining * (weight / total_weight)
        words.append({
            "word": token,
            "start": round(cursor, 3),
            "end": round(cursor + word_duration, 3),
        })
        cursor += word_duration

    # Guard against rounding drift
    if words:
        words[-1]["end"] = round(max(words[-1]["start"] + 0.01, end), 3)

    return words


def ensure_word_timestamps(
    segments: List[Dict[str, Any]],
    full_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Make sure every segment carries word-level timings.

    Transcripts produced without word timestamps (or older transcript files
    saved before word timings were supported) only have a segment start/end
    and the segment text.  Those segments are expanded here so the caption
    renderer always has something to work with.

    If the segments do not carry any text at all but a full transcript is
    available, the full text is spread across the segments proportionally
    to their duration first.

    Args:
        segments: Transcript segments
        full_text: Optional complete transcript text

    Returns:
        The same segments, with ``words`` populated where possible
    """
    import logging
    logger = logging.getLogger(__name__)

    segments = segments or []
    if not segments:
        return segments

    valid = [s for s in segments if isinstance(s, dict)]

    # Case 1: no word timings anywhere but every segment has text.
    needs_text_split = True
    if full_text:
        has_any_text = any((s.get("text") or "").strip() for s in valid)
        if not has_any_text:
            _spread_full_text_over_segments(valid, full_text)

    repaired = 0
    for segment in valid:
        words = segment.get("words")
        if words:
            needs_text_split = False
            continue

        text = segment.get("text") or ""
        if not text.strip():
            continue

        segment["words"] = synthesize_word_timestamps(
            text,
            segment.get("start", 0),
            segment.get("end", 0),
        )
        repaired += 1

    if repaired:
        logger.warning(
            "Repaired %d transcript segment(s) without word timestamps",
            repaired,
        )

    return segments


def _spread_full_text_over_segments(
    segments: List[Dict[str, Any]],
    full_text: str,
) -> None:
    """
    Distribute a full transcript across segments that have no text of their own.

    The words are handed out proportionally to each segment's duration, then
    each segment gets its own slice of text.
    """
    import logging
    logger = logging.getLogger(__name__)

    tokens = split_text_for_captions(full_text)
    if not tokens:
        return

    durations = []
    for segment in segments:
        try:
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", 0) or 0)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        durations.append(max(0.0, end - start))

    total_duration = sum(durations)
    if total_duration <= 0:
        # Fall back to an even split
        durations = [1.0] * len(segments)
        total_duration = float(len(segments))

    # How many words belong to each segment
    counts = [max(1, int(round(len(tokens) * (d / total_duration)))) for d in durations]

    # Fix rounding so the counts add up to the total number of words
    while sum(counts) > len(tokens):
        index = counts.index(max(counts))
        if counts[index] <= 1:
            break
        counts[index] -= 1
    while sum(counts) < len(tokens):
        index = durations.index(max(durations))
        counts[index] += 1

    position = 0
    for segment, count in zip(segments, counts):
        chunk = tokens[position:position + count]
        position += count
        if chunk:
            segment["text"] = " ".join(chunk)

    logger.info(
        "Spread %d transcript words across %d segments",
        len(tokens), len(segments),
    )
