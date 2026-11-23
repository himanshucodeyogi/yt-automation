"""
Phase 2: The Brain - AI Clip Extraction & SEO Generation

This module handles:
- Analyzing transcripts using Groq API with openai/gpt-oss-120b model
- Finding the most viral 30-60 second clip
- Generating viral Title, SEO Description, and Tags

Author: YouTube Shorts Automation Pipeline
"""

import os
import re
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

from pipeline_control import CancellationToken, PipelineCancelledError

try:
    from groq import Groq, RateLimitError, AuthenticationError, APIError
except ImportError:
    raise ImportError("groq is required. Install with: pip install groq")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
DEFAULT_MAX_TOKENS = 1024  # Compact limit prevents Groq TPM reservation overflow
DEFAULT_TEMPERATURE = 0.7

# Dangling conjunctions & transition tokens that should NEVER end a Short
HINDI_DANGLING_TOKENS = {
    "और", "तथा", "एवं", "व", "या", "अथवा",
    "कि", "क्योंकि", "चूंकि", "इसलिए", "अगर", "यदि", "तो",
    "लेकिन", "किन्तु", "परन्तु", "पर", "मगर", "बल्कि", "वरना", "ताकि",
    "जो", "जिसने", "जिसको", "जिससे", "जिसमें", "जहां", "जिधर", "जब", "जैसे", "वैसे",
    "मान", "माल", "लेते", "चलो", "मतलब", "यानी",
    "रहा", "रही", "रहे",
    # Hindi postpositions that cannot end a sentence
    "का", "के", "की", "को", "से", "में", "तक", "ने", "वाले", "वाली", "वाला"
}

ENGLISH_DANGLING_TOKENS = {
    "and", "or", "but", "nor", "so", "for", "yet",
    "because", "since", "although", "though", "even", "if", "unless",
    "while", "whereas", "as", "that", "which", "who", "whom", "whose",
    "when", "where", "after", "before", "until",
    "like", "suppose", "such",
    "to", "with", "in", "on", "at", "from", "by", "about", "into", "through",
    "a", "an", "the"
}

DANGLING_TOKENS = HINDI_DANGLING_TOKENS | ENGLISH_DANGLING_TOKENS

# Strong Hindi finite verb / predicate completion words
HINDI_SENTENCE_CLOSURES = {
    "है", "हैं", "हूँ", "हो", "था", "थी", "थे",
    "होगा", "होगी", "होंगे",
    "हुआ", "हुई", "हुए",
    "गया", "गई", "गए",
    "किया", "की", "किए",
    "दिया", "दी", "दिए",
    "लिया", "ली", "लिए",
    "आया", "आई", "आए",
    "पाया", "पाई", "पाए",
    "बना", "बनी", "बने",
    "चाहिए", "सकता", "सकती", "सकते",
    "पड़ेगा", "पड़ेगा", "पड़ेगी", "पड़ेंगे",
    "करो", "कीजिए", "दीजिए", "लीजिए", "लो", "बताओ", "देखो", "सुनो", "देगा", "देंगे"
}

# Number of words to include for context around timestamps
WORDS_PER_SECOND_CONTEXT = 2


@dataclass
class ClipRecommendation:
    """Recommended viral clip with SEO metadata."""
    clip_start_time: float  # Start time in seconds
    clip_end_time: float    # End time in seconds
    clip_text: str          # Text content of the clip
    viral_title: str        # Viral title for the Short
    seo_description: str    # SEO-optimized description
    tags: list              # List of relevant tags


class GroqAPIError(Exception):
    """Raised when Groq API call fails."""
    pass


class ClipExtractionError(Exception):
    """Raised when clip extraction fails."""
    pass


def clean_token(text: str) -> str:
    """Normalize token by stripping punctuation, preserving Unicode matras and ligatures."""
    return re.sub(r'[,.!?|।;:\'\"_()\[\]{}—\-]+', '', text).strip().lower()


def has_terminal_punct(text: str) -> bool:
    """Check if string ends with sentence-terminating punctuation."""
    return bool(re.search(r'[.!?|।]$', text.strip()))


def is_question_token(text: str) -> bool:
    """Check if string is an unanswered question."""
    return bool(re.search(r'\?$', text.strip()))


def refine_clip_boundaries(
    transcript_data: Dict[str, Any],
    raw_start: float,
    raw_end: float,
    min_duration: float = 28.0,
    max_duration: float = 58.0,
) -> Tuple[float, float, str]:
    """
    Refines raw LLM timestamps so the clip NEVER cuts off mid-sentence or mid-thought.
    
    Ensures:
    1. Start boundary starts cleanly at the beginning of a sentence/hook.
    2. End boundary snaps to a completed sentence, punchline, or takeaway.
    3. Trailing conjunctions / connectors ('और', 'कि', 'तो', 'and', etc.) are stripped.
    4. If a question is asked and its answer is delivered in the next few seconds,
       the boundary extends to include the answer rather than leaving the audience hanging.
    5. A natural speech tail padding is added so the final vowel finishes cleanly.
    6. Duration stays strictly in the optimal YouTube Shorts window (28s to 58s).
    """
    segments = transcript_data.get("segments", [])
    all_words: List[Dict[str, Any]] = []

    for s_idx, seg in enumerate(segments):
        words = seg.get("words", [])
        for w_idx, w in enumerate(words):
            raw_w = w.get("word", "").strip()
            if not raw_w:
                continue
            all_words.append({
                "word": raw_w,
                "clean": clean_token(raw_w),
                "start": float(w.get("start", 0)),
                "end": float(w.get("end", 0)),
                "seg_idx": s_idx,
                "is_seg_end": (w_idx == len(words) - 1),
                "seg_text": seg.get("text", "")
            })

    # Fallback if no word-level timestamps exist (segment-level snapping)
    if not all_words:
        if not segments:
            return raw_start, raw_end, ""
        
        # Snap start to closest segment start
        start_seg = min(segments, key=lambda s: abs(s.get("start", 0) - raw_start))
        refined_start = max(0.0, float(start_seg.get("start", raw_start)))
        
        # Snap end to closest segment end
        candidate_ends = [
            s for s in segments
            if (min_duration - 2) <= (float(s.get("end", 0)) - refined_start) <= (max_duration + 2)
        ]
        if candidate_ends:
            best_end_seg = min(candidate_ends, key=lambda s: abs(float(s.get("end", 0)) - raw_end))
            refined_end = float(best_end_seg.get("end", raw_end)) + 0.20
        else:
            refined_end = min(raw_end, refined_start + max_duration)
        
        return refined_start, refined_end, ""

    # --- START BOUNDARY REFINEMENT ---
    start_candidates = [
        w for w in all_words
        if (raw_start - 4.0) <= w["start"] <= (raw_start + 4.0)
    ]
    if not start_candidates:
        start_candidates = [min(all_words, key=lambda w: abs(w["start"] - raw_start))]

    best_start_word = min(start_candidates, key=lambda w: abs(w["start"] - raw_start))
    refined_start = max(0.0, best_start_word["start"] - 0.10)

    # --- END BOUNDARY REFINEMENT ---
    # Search window extends up to +15s to capture full answers, punchlines, and revelations
    end_candidates = [
        (idx, w) for idx, w in enumerate(all_words)
        if (raw_end - 8.0) <= w["end"] <= (raw_end + 15.0)
    ]
    if not end_candidates:
        end_candidates = [
            (idx, w) for idx, w in enumerate(all_words)
            if w["end"] >= (refined_start + min_duration)
        ]

    scored_candidates = []
    for idx, w in end_candidates:
        dur = w["end"] - refined_start
        can_shift_start = False
        needed_start = refined_start
        
        # If the narrative completion would exceed max_duration, check if we can shift start forward
        if dur > (max_duration + 1.0):
            for candidate_s in all_words:
                if (refined_start + 2.0) <= candidate_s["start"] <= (w["end"] - min_duration):
                    test_dur = w["end"] - candidate_s["start"]
                    if min_duration <= test_dur <= max_duration:
                        can_shift_start = True
                        needed_start = candidate_s["start"]
                        break
            if not can_shift_start:
                continue
        elif dur < 20.0:
            continue

        effective_dur = (w["end"] - needed_start) if can_shift_start else dur
        score = 0.0

        # Distance penalty from raw_end (mild so true closures win)
        dist = abs(w["end"] - raw_end)
        score -= dist * 6.0

        # Speech pause / silence gap after word
        next_w = all_words[idx + 1] if idx + 1 < len(all_words) else None
        gap = (next_w["start"] - w["end"]) if next_w else 1.5

        if gap >= 0.80:
            score += 250.0  # Clear sentence break / breathing pause
        elif gap >= 0.35:
            score += 150.0
        elif gap >= 0.18:
            score += 60.0

        has_term = has_terminal_punct(w["word"])
        is_closure = w["clean"] in HINDI_SENTENCE_CLOSURES

        # Decisive bonus for true sentence / dialogue completions
        if has_term and is_closure:
            score += 650.0
        elif has_term:
            score += 350.0
        elif is_closure:
            score += 140.0
        elif w["is_seg_end"]:
            score += 100.0

        # Dangling connector / conjunction penalty (NEVER end on these!)
        if w["clean"] in DANGLING_TOKENS:
            score -= 900.0
        elif w["word"].endswith(",") and not w["is_seg_end"] and gap < 0.40:
            score -= 200.0

        # Penalty for incomplete clause starters
        if idx > 0:
            prev_w = all_words[idx - 1]
            phrase = f"{prev_w['clean']} {w['clean']}"
            if phrase in ["मान लेते", "माल लेते", "है कि", "हैं कि", "हूँ कि", "बोलते हैं", "कहते हैं", "रहस्य बताया", "पूछा कि"]:
                score -= 900.0

        # Avoid ending on questions without answers if answers follow immediately
        if is_question_token(w["word"]):
            score -= 400.0

        # Penalize ending on an open subordinate clause (only if no terminal punctuation)
        if not has_term:
            clause_words = [all_words[j]["clean"] for j in range(max(0, idx - 10), idx + 1)]
            if "जब" in clause_words and "तब" not in clause_words:
                score -= 450.0
            if ("अगर" in clause_words or "यदि" in clause_words) and "तो" not in clause_words:
                score -= 450.0

        # Optimal Shorts duration bonus (30s - 55s)
        if 30.0 <= effective_dur <= 56.0:
            score += 100.0
        elif effective_dur > 58.0:
            score -= 400.0

        scored_candidates.append((score, idx, w, gap, can_shift_start, needed_start))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    if scored_candidates:
        best_score, best_idx, best_end_word, gap, can_shift_start, needed_start = scored_candidates[0]
        # Pad slightly into the pause so the final consonant/vowel doesn't clip
        pad = min(0.25, max(0.08, gap * 0.4))
        refined_end = best_end_word["end"] + pad
        if can_shift_start:
            refined_start = max(0.0, needed_start - 0.10)
    else:
        refined_end = min(raw_end, refined_start + max_duration)

    # Reconstruct text accurately
    clip_words = [
        w["word"] for w in all_words
        if w["start"] >= (refined_start - 0.05) and w["end"] <= (refined_end + 0.05)
    ]
    
    # Strip any trailing dangling tokens that might have sneaked in
    while clip_words and clean_token(clip_words[-1]) in DANGLING_TOKENS:
        clip_words.pop()
    
    clip_text = " ".join(clip_words)
    logger.info(
        f"[Brain] Boundary refined for sentence & story closure: "
        f"{raw_start:.2f}s-{raw_end:.2f}s -> {refined_start:.2f}s-{refined_end:.2f}s "
        f"(duration: {refined_end - refined_start:.2f}s)"
    )

    return refined_start, refined_end, clip_text


def _format_transcript_with_timestamps(transcript_data: Dict[str, Any]) -> str:
    """
    Reconstruct transcript with explicit START and END timestamp markers for EVERY complete sentence/utterance.
    
    Breaks multi-sentence chunks into individual sentences using punctuation (।, ?, !, .)
    and natural speech pauses (gap >= 0.65s) so the LLM can clearly see exactly where
    each thought, story hook, or punchline starts and finishes.
    
    Args:
        transcript_data: Transcript dictionary from Phase 1
    
    Returns:
        Formatted transcript string with precise timestamps
    """
    segments = transcript_data.get("segments", [])
    full_text = transcript_data.get("full_text", "")
    
    all_words = []
    for s_idx, seg in enumerate(segments):
        for w in seg.get("words", []):
            raw_w = w.get("word", "").strip()
            if raw_w:
                all_words.append({
                    "word": raw_w,
                    "clean": clean_token(raw_w),
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0)),
                    "seg_idx": s_idx
                })
                
    if not all_words:
        lines = []
        for segment in segments:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            text = segment.get("text", "").strip()
            if text:
                lines.append(f"[{start:.1f}s -> {end:.1f}s] {text}")
        return "\n".join(lines) if lines else full_text

    terminal_chars = set("।?!.")
    non_break_tokens = {"तुम", "आप", "वह", "वे", "ये", "यह", "और", "कि", "तो", "या", "जब", "अगर", "जैसे", "का", "के", "की", "को", "से", "में", "ने", "पर"}

    lines = []
    curr = []
    for i, w in enumerate(all_words):
        curr.append(w)
        next_w = all_words[i + 1] if i + 1 < len(all_words) else None
        
        has_punct = any(ch in w["word"] for ch in terminal_chars)
        pause = (next_w["start"] - w["end"]) if next_w else 1.5
        
        is_dangling = w["clean"] in non_break_tokens
        is_too_short = len(curr) < 3 and pause < 1.2
        is_closure = (w["clean"] in HINDI_SENTENCE_CLOSURES and pause >= 0.45)
        is_long = len(curr) >= 15 and pause >= 0.35
        
        should_break = False
        if next_w is None:
            should_break = True
        elif not is_dangling and not is_too_short:
            if has_punct or (pause >= 0.70) or is_closure or is_long:
                should_break = True
                
        if should_break:
            start = curr[0]["start"]
            end = curr[-1]["end"]
            text = " ".join(item["word"] for item in curr)
            lines.append(f"[{start:.1f}s -> {end:.1f}s] {text}")
            curr = []
            
    if curr:
        start = curr[0]["start"]
        end = curr[-1]["end"]
        text = " ".join(item["word"] for item in curr)
        lines.append(f"[{start:.1f}s -> {end:.1f}s] {text}")
        
    return "\n".join(lines)


def _build_system_prompt() -> str:
    """
    Build the system prompt for the Groq LLM.
    
    Defines the AI's role and strict instructions for viral content extraction
    and complete narrative closure (ending sense).
    
    Returns:
        System prompt string
    """
    return """
You are an expert YouTube Shorts viral content strategist with 10+ years of experience 
analyzing video content for maximum engagement, high retention, and complete narrative satisfaction.

Your task is to analyze the provided transcript to select the SINGLE MOST ENGAGING 30-58 SECOND segment.

============================================================
CRITICAL REQUIREMENTS FOR A SUCCESSFUL SHORT:
============================================================

1. HOOK (First 3 Seconds):
   - The clip MUST start cleanly at the very beginning of a sentence or powerful hook.
   - Look for curiosity-provoking questions, controversial opinions, shock, or high emotion.

2. STORY, REVELATION & NARRATIVE RESOLUTION (कहानी / रहस्य / बात का पूरा अंत - DO NOT CUT OFF):
   - A Short MUST feel like a complete, satisfying mini-story or resolved dialogue.
   - IF A MYSTERY, QUESTION, OR DRAMATIC PREMISE IS INTRODUCED:
     * Examples: "तुम्हारे पिता कौन हैं?", "अपने जन्म का रहस्य बताया", "रावण की लंका कहाँ थी?", "क्या हुआ था?"
     * YOU MUST INCLUDE THE COMPLETE ANSWER / REVELATION / CLIMAX BEFORE ENDING!
     * NEVER cut off after introducing a secret or while the explanation is underway (e.g., stopping when someone begins explaining their origin or the reason for an event leaves the audience furious and ruins the Short!).
   - SUBORDINATE / CONDITIONAL CLAUSES:
     * If a sentence has "जब..." (when...), you CANNOT end on that clause. The "तब..." (then...) resolution must finish!
     * If a sentence has "अगर..." (if...), the "तो..." outcome must finish!
   - STRICTLY FORBIDDEN: NEVER end the clip mid-sentence, mid-thought, or on a dangling conjunction!
   - NEVER end on words like:
     * Hindi: "और", "कि", "तो", "या", "लेकिन", "पर", "मगर", "क्योंकि", "अगर", "जैसे", "मान लेते हैं कि", "जब", "यानी", "का", "के", "में", "से"
     * English: "and", "but", "so", "because", "if", "when", "or", "that", "which", "like", "as", "to", "in"
   - RESOLUTION DURATION RULE (30s to 58s):
     * If including the entire preamble PLUS the resolution exceeds 58 seconds, DO NOT SACRIFICE THE ENDING!
     * Instead, MOVE THE START TIME FORWARD (start right at the core question, challenge, or hook) so that the entire dialogue and the full climax/answer fit inside 30-58 seconds!

3. TIMESTAMP PRECISION:
   - Each transcript line is formatted as: `[STARTs -> ENDs] text`
   - `clip_start_time`: Must be the `STARTs` value of the first sentence.
   - `clip_end_time`: Must be the `ENDs` value where the complete thought/story finishes.
   - Duration MUST be between 30 and 58 seconds (strictly enforced, never exceed 59 seconds).

OUTPUT FORMAT (STRICT JSON ONLY - no additional text):
{
  "clip_start_time": <float - start in seconds>,
  "clip_end_time": <float - end in seconds where thought finishes>,
  "clip_text": "<string - exact text in this segment ending with full stop or punctuation>",
  "viral_title": "<string - 40-60 chars, click-worthy, includes hook and emoji>",
  "seo_description": "<string - 1-2 compelling sentences with keywords and call-to-action>",
  "tags": ["<tag1>", "<tag2>", ..., "<tag5-10>"]
}

DO NOT output anything besides this JSON. No markdown, no commentary.
"""


def _build_user_prompt(transcript_with_timestamps: str, original_transcript: str = "") -> str:
    """
    Build the user prompt for the Groq LLM.
    
    Args:
        transcript_with_timestamps: Readable transcript with [MM:SS] markers
        original_transcript: Full text transcript (omitted when redundant to preserve TPM quota)
    
    Returns:
        User prompt string
    """
    # Notice: transcript_with_timestamps already contains the complete chronological text.
    # Omitting redundant duplication saves ~40% token budget, avoiding 413 / TPM rate limits.
    return f"""TRANSCRIPT WITH TIMESTAMPS:

{transcript_with_timestamps}

---

ANALYZE THIS TRANSCRIPT AND IDENTIFY THE SINGLE MOST VIRAL 30-58 SECOND CLIP.
Follow all narrative completeness, question-answer resolution, and JSON formatting rules specified in the system prompt.
Generate STRICT JSON output only.
"""


def _call_groq_api(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = 3,
    cancel_token: Optional[CancellationToken] = None,
) -> str:
    """
    Call Groq API with automatic model fallback and retry logic for TPM/rate limits.

    If a model hits tokens-per-minute (TPM) limits or 413 Payload Too Large,
    it automatically falls back to secondary models (e.g. llama-3.1-8b-instant, mixtral)
    without terminating the pipeline.
    """
    call_done = threading.Event()
    watcher = None

    if cancel_token is not None:
        def _close_client_on_cancel() -> None:
            while not call_done.wait(0.1):
                if cancel_token.is_cancelled():
                    try:
                        client.close()
                    except Exception:
                        pass
                    return

        watcher = threading.Thread(
            target=_close_client_on_cancel,
            name="groq-cancellation-watcher",
            daemon=True,
        )
        watcher.start()

    # Candidate models chain to prevent single-model TPM exhaustion
    models_to_try = [model]
    for fb in FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_error: Optional[Exception] = None

    try:
        for m_idx, current_model in enumerate(models_to_try):
            current_max_tokens = min(max_tokens, 1024)

            for attempt in range(max_retries):
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()

                try:
                    logger.info(f"Calling Groq API [{current_model}] (attempt {attempt + 1}/{max_retries})...")
                    
                    completion = client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=current_max_tokens,
                        temperature=temperature
                    )

                    if cancel_token is not None:
                        cancel_token.raise_if_cancelled()

                    response_text = completion.choices[0].message.content.strip()
                    logger.info(f"Groq API call successful with model [{current_model}]")
                    return response_text

                except PipelineCancelledError:
                    raise
                except AuthenticationError as e:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise PipelineCancelledError("Groq request cancelled by the user") from e
                    raise GroqAPIError(f"Invalid Groq API key: {e}") from e

                except (RateLimitError, APIError) as e:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise PipelineCancelledError("Groq request cancelled by the user") from e

                    err_msg = str(e).lower()
                    last_error = e

                    # Check for JSON truncation
                    if "json_validate_failed" in err_msg or "max completion tokens" in err_msg:
                        logger.warning(f"JSON validation cut off on [{current_model}], increasing max_tokens...")
                        current_max_tokens = min(current_max_tokens + 512, 2048)
                        continue

                    # Check if model is decommissioned / not found
                    is_model_unavailable = (
                        "decommissioned" in err_msg or 
                        "not found" in err_msg or 
                        "404" in err_msg or 
                        "does not exist" in err_msg
                    )

                    # Check if TPM (tokens per minute) limit or 413 payload too large hit
                    is_tpm_or_size = (
                        "tokens per minute" in err_msg or 
                        "limit 8000" in err_msg or 
                        "request too large" in err_msg or 
                        "413" in err_msg or 
                        "rate_limit_exceeded" in err_msg
                    )

                    if (is_model_unavailable or is_tpm_or_size) and m_idx < len(models_to_try) - 1:
                        next_model = models_to_try[m_idx + 1]
                        logger.warning(
                            f"⚠️ Model '{current_model}' unavailable or hit token/TPM limit ({e}). "
                            f"Automatically switching to fallback model '{next_model}'..."
                        )
                        break  # Break inner loop to try next model immediately

                    wait_time = min(2 ** attempt, 8)
                    logger.warning(f"Groq API retry on [{current_model}], waiting {wait_time}s (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        if cancel_token is not None:
                            if cancel_token.wait(wait_time):
                                raise PipelineCancelledError("Groq request cancelled by the user")
                        else:
                            time.sleep(wait_time)
                    else:
                        if m_idx < len(models_to_try) - 1:
                            logger.warning(f"Exhausted retries on [{current_model}], falling back to next model...")
                            break

                except Exception as e:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise PipelineCancelledError("Groq request cancelled by the user") from e
                    raise GroqAPIError(f"Unexpected error calling Groq API: {e}") from e

        raise GroqAPIError(f"Groq API error across all models: {last_error}")
    finally:
        call_done.set()


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Safely parse JSON response from LLM.
    
    Args:
        response_text: Raw response text
    
    Returns:
        Parsed JSON dictionary
    
    Raises:
        ClipExtractionError: If JSON parsing fails
    """
    try:
        # Try to find JSON in the response (in case of extra text)
        # Look for JSON object pattern
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ClipExtractionError(f"No JSON found in response: {response_text[:200]}...")
        
        json_str = response_text[json_start:json_end]
        parsed = json.loads(json_str)
        
        # Validate required fields
        required_fields = ["clip_start_time", "clip_end_time", "clip_text", 
                          "viral_title", "seo_description", "tags"]
        
        missing = [f for f in required_fields if f not in parsed]
        if missing:
            raise ClipExtractionError(f"Missing required fields: {missing}")
        
        # Validate types
        if not isinstance(parsed["clip_start_time"], (int, float)):
            raise ClipExtractionError("clip_start_time must be a number")
        if not isinstance(parsed["clip_end_time"], (int, float)):
            raise ClipExtractionError("clip_end_time must be a number")
        if not isinstance(parsed["tags"], list):
            raise ClipExtractionError("tags must be a list")
        
        # Initial sanity check on duration (detailed sentence-boundary refinement happens next)
        duration = parsed["clip_end_time"] - parsed["clip_start_time"]
        if duration <= 5.0:
            parsed["clip_end_time"] = parsed["clip_start_time"] + 35.0
        
        return parsed
        
    except json.JSONDecodeError as e:
        raise ClipExtractionError(f"Failed to parse JSON response: {e}\nResponse: {response_text[:500]}...")


def analyze_transcript_for_clip(
    transcript_data: Dict[str, Any],
    groq_api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """
    Analyze transcript to find viral clip and generate SEO metadata.
    
    This is the main Phase 2 function. It sends the transcript to Groq
    using the openai/gpt-oss-120b model and returns the optimal clip
    recommendation with viral SEO metadata.
    
    Args:
        transcript_data: Transcript dictionary from Phase 1
        groq_api_key: Groq API key
        model: Groq model to use (default: openai/gpt-oss-120b)
        temperature: Sampling temperature (default: 0.7)
        cancel_token: Optional token used to interrupt the analysis
    
    Returns:
        Dict containing:
            - clip_start_time: Optimal start time in seconds
            - clip_end_time: Optimal end time in seconds  
            - clip_text: Text content of the recommended clip
            - viral_title: Click-worthy title for the Short
            - seo_description: SEO-optimized description
            - tags: List of relevant tags
    
    Raises:
        GroqAPIError: If API call fails
        ClipExtractionError: If clip extraction or parsing fails
    """
    # Validate API key
    if not groq_api_key or groq_api_key.strip() == "":
        raise GroqAPIError("Groq API key is required. Set GROQ_API_KEY environment variable.")
    
    logger.info("="*60)
    logger.info("PHASE 2: The Brain - AI Clip Extraction")
    logger.info("="*60)
    
    # Step 1: Format transcript with timestamps
    logger.info("\n--- STEP 1: Formatting transcript with timestamps ---")
    transcript_with_timestamps = _format_transcript_with_timestamps(transcript_data)
    full_text = transcript_data.get("full_text", "")
    
    logger.info(f"Formatted transcript length: {len(transcript_with_timestamps)} chars")
    logger.info(f"Full text length: {len(full_text)} chars")
    
    # Step 2: Build prompts
    logger.info("\n--- STEP 2: Building LLM prompts ---")
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(transcript_with_timestamps, full_text)
    
    # Step 3: Call Groq API
    logger.info("\n--- STEP 3: Calling Groq API ---")

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        raise GroqAPIError(f"Failed to initialize Groq client: {e}") from e
    
    response_text = _call_groq_api(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=temperature,
        cancel_token=cancel_token,
    )

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    
    # Step 4: Parse response
    logger.info("\n--- STEP 4: Parsing LLM response ---")
    
    try:
        result = _parse_json_response(response_text)
    except ClipExtractionError:
        logger.error(f"Raw response: {response_text[:500]}...")
        raise
    
    # Step 5: Refine clip boundaries for complete sentence closure & narrative sense
    logger.info("\n--- STEP 5: Refining clip boundaries for complete sentence & narrative closure ---")
    raw_start = float(result["clip_start_time"])
    raw_end = float(result["clip_end_time"])
    
    refined_start, refined_end, refined_text = refine_clip_boundaries(
        transcript_data=transcript_data,
        raw_start=raw_start,
        raw_end=raw_end,
        min_duration=28.0,
        max_duration=58.0,
    )
    
    result["clip_start_time"] = refined_start
    result["clip_end_time"] = refined_end
    if refined_text:
        result["clip_text"] = refined_text

    # Log results
    logger.info("\n--- RESULTS ---")
    logger.info(f"Clip duration: {result['clip_end_time'] - result['clip_start_time']:.1f}s")
    logger.info(f"Start time: {result['clip_start_time']:.2f}s")
    logger.info(f"End time: {result['clip_end_time']:.2f}s")
    logger.info(f"Title: {result['viral_title'][:60]}...")
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 COMPLETE - Clip recommendation ready")
    logger.info("="*60)
    
    return {
        "clip_start_time": float(result["clip_start_time"]),
        "clip_end_time": float(result["clip_end_time"]),
        "clip_text": str(result["clip_text"]),
        "viral_title": str(result["viral_title"]),
        "seo_description": str(result["seo_description"]),
        "tags": list(result["tags"]),
        "is_boundary_refined": True,
    }


def run_phase2_pipeline(
    transcript_path: str,
    groq_api_key: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 2 pipeline.
    
    Loads a transcript from Phase 1 and analyzes it for viral content.
    
    Args:
        transcript_path: Path to transcript JSON file from Phase 1
        groq_api_key: Groq API key (or set GROQ_API_KEY env var)
    
    Returns:
        Clip recommendation with SEO metadata
    
    Example:
        >>> result = run_phase2_pipeline("downloads/abc123_transcript.json")
        >>> print(result['viral_title'])
    """
    # Load transcript
    logger.info(f"Loading transcript from: {transcript_path}")
    
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Get API key
    if groq_api_key is None:
        groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError(
            "Groq API key required. Either pass groq_api_key parameter "
            "or set GROQ_API_KEY environment variable."
        )
    
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    return analyze_transcript_for_clip(
        transcript_data,
        groq_api_key,
        cancel_token=cancel_token,
    )


# CLI Entry Point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 2: AI Clip Extraction & SEO Generation"
    )
    parser.add_argument(
        "transcript_file",
        help="Path to transcript JSON file from Phase 1"
    )
    parser.add_argument(
        "--api-key",
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_phase2_pipeline(
            transcript_path=args.transcript_file,
            groq_api_key=args.api_key
        )
        
        print("\n" + "="*60)
        print("CLIP RECOMMENDATION:")
        print("="*60)
        print(f"\nTimestamp: {result['clip_start_time']:.1f}s - {result['clip_end_time']:.1f}s")
        print(f"Duration: {result['clip_end_time'] - result['clip_start_time']:.1f}s")
        print(f"\nClip Text Preview: {result['clip_text'][:100]}...")
        
        print("\n" + "="*60)
        print("SEO METADATA:")
        print("="*60)
        print(f"\n📌 Title: {result['viral_title']}")
        print(f"\n📝 Description: {result['seo_description']}")
        print(f"\n🏷️ Tags: {', '.join(result['tags'])}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit(1)
    except (GroqAPIError, ClipExtractionError) as e:
        logger.error(f"Processing failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)