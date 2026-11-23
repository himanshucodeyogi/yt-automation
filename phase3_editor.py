"""
Phase 3: The Editor - Video Cutting & Animated Captions

This module handles:
- Trimming video to exact clip timestamps
- Center-cropping to 9:16 aspect ratio (YouTube Shorts)
- Word-by-word animated captions with Hormozi styling
- Composite and export with proper codecs

Author: YouTube Shorts Automation Pipeline
"""

import os
import sys
import time
import json
import logging
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from proglog import ProgressBarLogger

from pipeline_control import (
    CancellationToken,
    PipelineCancelledError,
    ensure_word_timestamps,
)

# Add imageio-ffmpeg to PATH before importing moviepy
import imageio_ffmpeg
imageio_ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if os.path.exists(imageio_ffmpeg_dir):
    os.environ['PATH'] = imageio_ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

import ffmpeg
import numpy as np
from moviepy import (
    ImageClip,
    CompositeVideoClip,
    VideoFileClip,
    AudioFileClip,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_FILENAME = "final_short.mp4"

# Hormozi-style caption styling
# Arial/DejaVu have much better Unicode (Hindi/Devanagari etc.) support than Impact
CAPTION_FONT = "Arial"  # Bold, sans-serif font with good Unicode support
CAPTION_FONT_SIZE = 40  # Clean, readable size for multi-word lines
CAPTION_COLOR = "#FFE600"  # Vibrant Yellow for active (spoken) word
CAPTION_INACTIVE_COLOR = "#FFFFFF"  # Crisp White for normal (unspoken) words
CAPTION_STROKE_COLOR = "black"
CAPTION_STROKE_WIDTH = 4  # Crisp stroke for visibility on any background
CAPTION_POSITION = "below"  # "below" (under card), "above" (header above card), "inside" (lower third), "middle"

# Video presentation layout:
# "card": Centered video box with rounded borders on pitch-black 9:16 background (reference style)
# "full": Full-screen 9:16 vertical crop
DEFAULT_VIDEO_LAYOUT = "card"

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

CARD_BOX_WIDTH = 1000
CARD_BOX_HEIGHT = 1100
CARD_CORNER_RADIUS = 32
CARD_BOX_X = (CANVAS_WIDTH - CARD_BOX_WIDTH) // 2   # 40
CARD_BOX_Y = (CANVAS_HEIGHT - CARD_BOX_HEIGHT) // 2  # 410
CARD_CAPTION_FONT_SIZE = 50
CARD_CAPTION_MARGIN = 20  # 20px margin between caption and video clip box
CARD_FOOTER_FONT_SIZE = 36
CARD_FOOTER_COLOR = "#FF3B30"

# Exactly 30px margin from the bottom of the video frame as requested
CAPTION_BOTTOM_MARGIN_PX = 30
CAPTION_BOTTOM_MARGIN_RATIO = 0.028  # ~2.8% of 1080p frame height (30px)
CAPTION_BOTTOM_MARGIN = 30  # 30px fallback

# Captions are positioned by their BASELINE (the line the letters sit on)
# rather than their top edge.  That keeps the text perfectly steady when the
# highlighted word changes, even if one word has a descender and another not.
# Calibrated so the lowest pixel of the letters + stroke + glow ends at exactly 30px
# from the bottom edge of the frame.
CAPTION_BASELINE_RATIO = 0.055

# Bundled Unicode fonts (Noto) - guarantee Devanagari/Unicode caption rendering.
# Windows usually ships no Devanagari-capable font, which makes captions
# render as empty "tofu" boxes.
ASSETS_FONTS_DIR = Path("assets") / "fonts"
BUNDLED_FONT = ASSETS_FONTS_DIR / "NotoSansDevanagari-Bold.ttf"
BUNDLED_FONT_FALLBACK = ASSETS_FONTS_DIR / "NotoSans-Bold.ttf"

# Timing constants
WORD_OVERLAP_BUFFER = 0.1  # Seconds of overlap for smooth transitions
WORD_FADE_DURATION = 0.2  # Fade in/out duration for each word
CAPTION_MAX_HOLD = 0.8    # Keep a line visible up to this long while waiting for the next one

# Multi-word karaoke captions: words AUTO-FILL each line dynamically based on screen width
# and speech flow (no fixed 4-word constraint). The word currently being spoken glows and
# grows ~28% bigger in vibrant yellow, while the other words stay normal size in white.
# Once spoken, it shrinks back to normal white and the next word lights up.
CAPTION_GROUP_SIZE = 6          # Default target words per line when auto-filling
CAPTION_ACTIVE_SCALE = 1.28     # How much bigger the spoken word is (28%)
CAPTION_WORD_SPACING = 0.20     # Gap between words (fraction of font size)
CAPTION_GLOW_RADIUS = 0.22      # Glow radius (fraction of active font size)
CAPTION_GLOW_COLOR = "#FFE600"  # Warm vibrant yellow halo behind active word
CAPTION_GROUP_STROKE_WIDTH = 4  # Outline thickness for crisp readability
CAPTION_MAX_WIDTH_RATIO = 0.88  # Line uses up to 88% of video width (6% safe margin on each side)

# Video codecs
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


class CaptionGenerationError(Exception):
    """Raised when caption generation fails."""
    pass


class _CancellableProgressLogger(ProgressBarLogger):
    """MoviePy logger that checks cancellation before every exported frame."""

    def __init__(self, cancel_token: CancellationToken) -> None:
        super().__init__(logged_bars=None)
        self._cancel_token = cancel_token

    def callback(self, **kwargs) -> None:
        self._cancel_token.raise_if_cancelled()


def _run_ffmpeg(command, cancel_token: Optional[CancellationToken] = None):
    """Run an ffmpeg-python command with immediate process cancellation."""
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    process = command.run_async(quiet=True)

    if cancel_token is not None:
        cancel_token.register_process(process)

    try:
        if cancel_token is None:
            stdout, stderr = process.communicate()
        else:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    cancel_token.raise_if_cancelled()

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        if process.returncode != 0:
            raise ffmpeg.Error("ffmpeg", stdout, stderr)

        return stdout, stderr

    except PipelineCancelledError:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
        except OSError:
            pass
        raise
    finally:
        if cancel_token is not None:
            cancel_token.unregister_process(process)


def _run_subprocess(cmd: List[str], cancel_token: Optional[CancellationToken] = None):
    """Run an external command with immediate process cancellation."""
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cancel_token is not None:
        cancel_token.register_process(proc)

    try:
        stdout, stderr = proc.communicate()
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            raise VideoProcessingError(f"FFmpeg command failed ({proc.returncode}): {err}")
        return stdout, stderr
    except PipelineCancelledError:
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        raise
    finally:
        if cancel_token is not None:
            cancel_token.unregister_process(proc)


def _ensure_output_dir() -> Path:
    """Ensure output directory exists."""
    output_path = OUTPUT_DIR.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_pil_font(font) -> Optional[str]:
    """
    Resolve a font (path or name) to a real font file that PIL can load.
    
    PIL needs an actual TTF/OTF file - it cannot use ImageMagick style font
    names.  The bundled Noto fonts are always preferred, then the Windows
    font directory.
    
    Args:
        font: Font path or font name
        
    Returns:
        Path to a font file, or None when nothing could be resolved
    """
    if not font:
        return None

    if os.path.isfile(font):
        return font

    # A font name was given - look for a matching file.
    wanted = os.path.basename(str(font)).lower().replace(" ", "")
    wanted = os.path.splitext(wanted)[0]

    search_dirs = [
        BUNDLED_FONT.parent,
        Path(r"C:\Windows\Fonts"),
    ]

    for directory in search_dirs:
        if not directory.exists():
            continue
        try:
            for candidate in directory.iterdir():
                if not candidate.is_file():
                    continue
                name = candidate.stem.lower().replace(" ", "")
                if wanted and wanted in name:
                    return str(candidate)
        except OSError:
            continue

    # Last resort - the bundled fonts are guaranteed to exist.
    for fallback in (BUNDLED_FONT, BUNDLED_FONT_FALLBACK):
        if fallback.exists():
            return str(fallback)

    return None


def _render_probe_bitmap(text: str, font, font_size: int = 32):
    """
    Render text to a raw bitmap so two renders can be compared.
    
    PIL is used instead of MoviePy's TextClip because MoviePy computes the
    canvas height from font metrics that do not always match what is really
    drawn (the bottom of the glyphs gets clipped for some fonts).
    
    Returns:
        Tuple of (width, height, raw_bytes) or None if rendering failed
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_path = _resolve_pil_font(font)
        if not font_path:
            return None

        pil_font = ImageFont.truetype(font_path, font_size)
        ascent, descent = pil_font.getmetrics()

        width = max(8, int(pil_font.getlength(text)) + 40)
        height = max(8, ascent + descent + 40)

        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20 + ascent), text, font=pil_font, fill=255, anchor="ls")

        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        return (img.size[0], img.size[1], img.tobytes())
    except Exception:
        return None


def _font_has_glyph(font, char: str) -> Optional[bool]:
    """
    Check whether a font has a real glyph for a character.
    
    A missing glyph is rendered as a "tofu" box. We detect that by comparing
    the character against two Private Use Area codepoints (which no font has
    a glyph for) - if all three render identically, the font lacks the glyph.
    
    Args:
        font: Font name or path
        char: Single character to test
        
    Returns:
        True = has real glyph, False = tofu box, None = could not determine
    """
    target = _render_probe_bitmap(char, font)
    tofu_a = _render_probe_bitmap("\ue001", font)
    tofu_b = _render_probe_bitmap("\ue002", font)
    
    if target is None or tofu_a is None or tofu_b is None:
        return None
    
    # The two PUA chars must render identically to serve as a tofu reference
    if tofu_a[0] != tofu_b[0] or tofu_a[1] != tofu_b[1] or tofu_a[2] != tofu_b[2]:
        return None
    
    if target[0] != tofu_a[0] or target[1] != tofu_a[1]:
        return True  # different dimensions -> a real glyph was drawn
    
    return target[2] != tofu_a[2]  # different pixels -> a real glyph


def _caption_font_sample(words: List[Dict]) -> str:
    """
    Build a sample string representing the characters captions must render.
    
    Uses the actual non-ASCII characters from the transcript so the font test
    reflects what will really be drawn.
    """
    chars = set()
    for word_data in words:
        for ch in word_data.get("word", ""):
            if ord(ch) > 127:
                chars.add(ch)
    
    if chars:
        return "".join(sorted(chars))[:8]
    return "ABC"


def _get_available_fonts(
    sample_text: str = "ABC",
    cancel_token: Optional[CancellationToken] = None,
) -> Optional[str]:
    """
    Find a font that can render the given text WITHOUT tofu boxes.
    
    Windows ships almost no Devanagari-capable font, so system defaults
    (Arial, Segoe UI, Impact, MoviePy's default) render Hindi captions as
    empty boxes. We therefore prefer the bundled Noto fonts and verify every
    candidate by actually rendering it.
    
    Args:
        sample_text: Text that must be renderable (e.g. caption characters)
        
    Returns:
        Font name/path to use, or None to fall back to MoviePy's default
    """
    # Characters we need the font to support (take one representative char)
    test_char = next((c for c in sample_text if ord(c) > 127), sample_text[:1] or "A")
    
    # Bundled Noto fonts first - these are guaranteed Unicode/Devanagari capable
    candidates = []
    if BUNDLED_FONT.exists():
        candidates.append(str(BUNDLED_FONT))
    if BUNDLED_FONT_FALLBACK.exists():
        candidates.append(str(BUNDLED_FONT_FALLBACK))
    
    # System fonts with wide Unicode coverage (if present)
    candidates += [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",
        r"C:\Windows\Fonts\ARIALUNI.TTF",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    
    # Font names resolved through ImageMagick
    candidates += [
        "Noto Sans Devanagari",
        "Arial Unicode MS",
        "DejaVu Sans",
        "FreeSans",
        "Nirmala UI",
        "Mangal",
        "Arial",
        "Impact",
    ]
    
    for font in candidates:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        # Skip absolute paths that do not exist
        if os.path.isabs(font) and not os.path.exists(font):
            continue
        
        try:
            status = _font_has_glyph(font, test_char)
            if status is True:
                logger.info(f"Caption font selected: {font} (renders '{test_char}' correctly)")
                return font
            if status is False:
                logger.warning(f"Font '{font}' shows tofu boxes for '{test_char}', skipping")
        except Exception as e:
            logger.debug(f"Font test failed for {font}: {e}")
            continue
    
    logger.warning("No font found that renders the caption text cleanly; using default")
    return None


def _crop_to_short_ratio(
    video_path: str,
    output_path: str,
    cancel_token: Optional[CancellationToken] = None,
) -> str:
    """
    Use ffmpeg to center-crop video from 16:9 to 9:16.
    
    Args:
        video_path: Input video path
        output_path: Output video path
    
    Returns:
        Path to cropped video
    """
    logger.info(f"Center-cropping video to 9:16: {video_path}")
    
    try:
        # Explicitly map video + audio: -filter_complex disables ffmpeg's
        # automatic stream selection, which would otherwise drop the audio.
        crop_input = ffmpeg.input(video_path)
        _run_ffmpeg(
            (
                ffmpeg
                .output(
                    crop_input.video
                        .filter('scale', 1080, 1920)  # Standard Shorts resolution
                        .filter('crop', 1080, 1920),  # Ensure exact crop
                    crop_input.audio,
                    output_path,
                    vcodec=VIDEO_CODEC,
                    acodec=AUDIO_CODEC,
                    movflags='+faststart'
                )
                .overwrite_output()
            ),
            cancel_token,
        )
        
        logger.info(f"Cropped video saved to: {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg crop error: {e}")
        raise VideoProcessingError(f"Failed to crop video: {e}") from e


def _has_audio_stream(
    video_path: str,
    cancel_token: Optional[CancellationToken] = None,
) -> bool:
    """
    Check if a video file contains an audio stream.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if audio stream exists, False otherwise
    """
    try:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        probe = ffmpeg.probe(video_path)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        audio_streams = [s for s in probe['streams'] if s.get('codec_type') == 'audio']
        return len(audio_streams) > 0
    except PipelineCancelledError:
        raise
    except Exception as e:
        logger.warning(f"Could not probe audio stream for {video_path}: {e}")
        return False


def _ensure_audio_in_output(
    final_output_path: str,
    audio_source_path: str,
    cancel_token: Optional[CancellationToken] = None,
) -> None:
    """
    If the exported output video has no audio stream, remux audio from the source.
    
    This ensures the final Short always has voice/audio even if MoviePy drops it.
    
    Args:
        final_output_path: Path to the exported final video (may lack audio)
        audio_source_path: Path to a video with audio to copy from
    """
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    if _has_audio_stream(final_output_path, cancel_token):
        logger.info("Final output already has audio stream")
        return
    
    logger.warning("Final output has no audio stream - remuxing audio from source video")
    
    # Build a temp path for the remuxed file
    output_path = Path(final_output_path)
    temp_path = str(output_path.with_suffix('.tmp.mp4'))
    
    try:
        video_input = ffmpeg.input(final_output_path).video
        audio_input = ffmpeg.input(audio_source_path).audio

        # Probe video duration to apply a smooth ending audio fade-out
        # This eliminates any abrupt cut, pop, or click at the end of the Short.
        duration = 0.0
        try:
            probe = ffmpeg.probe(final_output_path)
            duration = float(probe.get('format', {}).get('duration', 0))
            if duration <= 0:
                v_stream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None)
                if v_stream:
                    duration = float(v_stream.get('duration', 0))
        except Exception:
            duration = 0.0

        if duration > 1.0:
            fade_start = max(0.0, duration - 0.35)
            audio_out = audio_input.filter('afade', type='out', start_time=fade_start, duration=0.35)
        else:
            audio_out = audio_input

        _run_ffmpeg(
            (
                ffmpeg
                .output(
                    video_input,
                    audio_out,
                    temp_path,
                    vcodec='copy',
                    acodec='aac',
                    audio_bitrate='192k',
                    movflags='+faststart',
                )
                .overwrite_output()
            ),
            cancel_token,
        )
        
        # Replace the original with the remuxed version
        import shutil
        shutil.move(temp_path, final_output_path)
        logger.info(f"Audio remuxed successfully into: {final_output_path}")
        
    except PipelineCancelledError:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise
    except ffmpeg.Error as e:
        logger.error(f"Failed to remux audio: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        logger.error(f"Unexpected error while remuxing audio: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _filter_words_for_clip(
    transcript_words: List[Dict],
    clip_start: float,
    clip_end: float,
    full_text: Optional[str] = None,
) -> List[Dict]:
    """
    Filter word-level transcript to only include words within the clip timeframe.
    
    Args:
        transcript_words: List of word dicts from Phase 1 transcript
        clip_start: Clip start time (absolute timestamp)
        clip_end: Clip end time (absolute timestamp)
        full_text: Optional complete transcript text, used to rebuild the
            segment text when a transcript only has segment timings
    
    Returns:
        List of words with adjusted (relative) timestamps
    """
    filtered_words = []
    
    # Older transcripts (or transcripts from a back-end that only returns
    # segment timestamps) have no word-level timings.  Repair them here so
    # the captions are never empty.
    ensure_word_timestamps(transcript_words, full_text)
    
    for segment in transcript_words:
        for word_data in segment.get("words", []):
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", 0)
            
            # Skip words entirely before or after the clip
            if word_end < clip_start or word_start > clip_end:
                continue
            
            # Adjust timestamps to be relative to clip start
            relative_start = max(0, word_start - clip_start)
            relative_end = max(0, word_end - clip_start)

            # Clamp single word duration to realistic limit (prevents stuck captions from Whisper glitches)
            word_str = word_data.get("word", "")
            max_word_len = max(0.35, min(1.2, len(word_str) * 0.12))
            if (relative_end - relative_start) > max_word_len or relative_end <= relative_start:
                relative_end = relative_start + max_word_len

            filtered_words.append({
                "word": word_str,
                "start": relative_start,
                "end": relative_end
            })
    
    # Sort by start time to ensure correct order
    filtered_words.sort(key=lambda w: w["start"])
    
    logger.info(f"Filtered {len(filtered_words)} words for clip duration {clip_end - clip_start:.2f}s")
    return filtered_words


HINDI_SPELLING_CORRECTIONS = {
    "धरम": "धर्म",
    "करम": "कर्म",
    "दुशकर्म": "दुष्कर्म",
    "पर्मो": "परमो",
    "धर्मा": "धर्म",
    "सुक": "सुख",
    "बिल्ता": "मिलता",
    "दरिदरता": "दरिद्रता",
    "अशान्ती": "अशांति",
    "पोवित्र": "पवित्र",
    "निन्दा": "निंदा",
    "ग्यान": "ज्ञान",
    "ग्यानी": "ज्ञानी",
    "आत्मा": "आत्मा",
    "शांती": "शांति",
    "क्रोध": "क्रोध",
    "अहकार": "अहंकार",
    "ससार": "संसार",
    "ससृति": "संस्कृति",
    "सपादक": "संपादक",
    "सबध": "संबंध",
    "सुन्दर": "सुंदर",
    "सतो": "संतों",
    "भगवन": "भगवान",
    "भगवत": "भागवत",
    "प्रभू": "प्रभु",
    "लक्षमन": "लक्ष्मण",
    "रामायण": "रामायण",
    "ह्रदय": "हृदय",
    "क्रपा": "कृपा",
    "पिरती": "प्रीति",
    "मुक्ती": "मुक्ति",
    "भक्ती": "भक्ति",
    "सतय": "सत्य",
    "अहसा": "अहिंसा",
    "अहसिा": "अहिंसा",
    "ब्रम्ह": "ब्रह्म",
    "कस्ट": "कष्ट",
    "स्रष्टि": "सृष्टि",
    "ऋशी": "ऋषि",
    "मनि": "मुनि",
    "पुन्य": "पुण्य",
    "कल्यान": "कल्याण",
    "परनाम": "प्रणाम",
    "चरन": "चरण",
    "आसीर्वाद": "आशीर्वाद",
    "आशीर्बाद": "आशीर्वाद",
}


def _correct_hindi_word(word: str) -> str:
    """Normalize common phonetic Speech-to-Text misspellings in Hindi."""
    if not word:
        return ""
    import re
    cleaned = word.strip()
    m = re.match(r"^([^\w\s]*)(.*?)([^\w\s]*)$", cleaned, re.UNICODE)
    if not m:
        return cleaned
    prefix, core, suffix = m.groups()
    if core in HINDI_SPELLING_CORRECTIONS:
        core = HINDI_SPELLING_CORRECTIONS[core]
    return prefix + core + suffix


def _sanitize_caption_text(text: str) -> str:
    """
    Clean transcript text so it can actually be rendered as a caption.
    Preserves Hindi Zero-Width Joiner (ZWJ) and Non-Joiner (ZWNJ) for ligatures.
    """
    if not text:
        return ""

    # Allow zero-width joiners (\u200c, \u200d) essential for Hindi conjuncts
    cleaned = "".join(
        ch for ch in text
        if ch != "\ufffd"
        and (unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs", "Cn") or ch in ("\u200c", "\u200d"))
    )

    cleaned = cleaned.strip()
    # Correct phonetic STT Hindi misspellings
    cleaned = _correct_hindi_word(cleaned)
    return cleaned


def _rgba(color: str) -> Tuple[int, int, int, int]:
    """Convert a color name / hex string into an RGBA tuple."""
    try:
        from PIL import ImageColor

        r, g, b = ImageColor.getrgb(color)
        return (r, g, b, 255)
    except Exception:
        return (255, 255, 255, 255)


def _render_caption_bitmap(
    text: str,
    font,
    font_size: int,
    color: str = CAPTION_COLOR,
    stroke_color: str = CAPTION_STROKE_COLOR,
    stroke_width: int = CAPTION_STROKE_WIDTH,
):
    """
    Render a caption word into an RGBA bitmap with a consistent baseline.
    
    MoviePy's TextClip derives the canvas size from font metrics that do not
    always match the glyphs it actually draws - with the bundled Noto fonts
    the bottom of the letters was cut off ("WILL" looked like "WT| |").
    Rendering with PIL avoids that completely.
    
    Every word is drawn on a canvas of the SAME height with the text on the
    SAME baseline.  That way the caption does not jump up and down when the
    highlighted word changes (a word with a descender like "DISPATCH," would
    otherwise sit lower than one without).
    
    Args:
        text: The text to render
        font: Font path or name
        font_size: Font size in pixels
        color: Fill colour
        stroke_color: Outline colour
        stroke_width: Outline thickness in pixels
        
    Returns:
        A PIL RGBA image, or None when rendering failed
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_path = _resolve_pil_font(font)
        if not font_path:
            return None

        pil_font = ImageFont.truetype(font_path, font_size)
        ascent, descent = pil_font.getmetrics()

        sw = max(0, int(stroke_width))
        pad_x = sw + 4

        # Fixed canvas height / baseline for every word at this font size.
        # The extra stroke allowance keeps descenders and their outline
        # inside the canvas.
        height = max(4, ascent + descent + (3 * sw) + 8)
        baseline = ascent + sw + 4

        width = max(4, int(pil_font.getlength(text)) + 2 * pad_x)

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (pad_x, baseline),
            text,
            font=pil_font,
            fill=_rgba(color),
            stroke_width=sw,
            stroke_fill=_rgba(stroke_color),
            anchor="ls",
        )

        # Crop horizontally only - the canvas height (and therefore the
        # baseline) must stay identical for every word.
        bbox = img.getchannel("A").getbbox()
        if bbox:
            img = img.crop((bbox[0], 0, bbox[2], height))

        return img
    except Exception as e:
        logger.warning(f"Failed to render caption '{text}': {e}")
        return None


def _fit_caption_font_size(
    text: str,
    font: Optional[str],
    max_width: int,
    base_size: int = CAPTION_FONT_SIZE,
) -> int:
    """
    Shrink the caption font so a long word still fits on one line.
    
    Without this, a very long word wraps to 3 lines and pushes the caption
    all the way down to (or past) the bottom edge of the frame.
    
    Args:
        text: The caption text
        font: Font path (or name) - a path allows exact measurement
        max_width: Maximum usable width in pixels
        base_size: Font size to start from
        
    Returns:
        A font size that fits the text inside ``max_width``
    """
    if not text:
        return base_size
    
    usable = max(40, int(max_width * 0.96))
    
    # Exact measurement when we have a real font file
    if font and os.path.isfile(font):
        try:
            from PIL import ImageFont
            image_font = ImageFont.truetype(font, base_size)
            bbox = image_font.getbbox(text)
            # The outline is drawn outside the glyph, so it adds width too.
            width = (bbox[2] - bbox[0]) + (CAPTION_STROKE_WIDTH * 2)
            if width > usable:
                scaled = int(base_size * (usable / width))
                return max(20, min(base_size, scaled))
            return base_size
        except Exception:
            pass
    
    # Rough estimate for named fonts (avg bold sans glyph ~= 0.58 * size)
    estimated = len(text) * base_size * 0.58 + (CAPTION_STROKE_WIDTH * 2)
    if estimated > usable:
        return max(20, int(base_size * (usable / estimated)))
    return base_size


def _create_rounded_mask_file(box_w: int, box_h: int, radius: int, out_path: str) -> str:
    """
    Create a high-resolution antialiased rounded rectangle mask PNG for FFmpeg.
    Used to frame videos cleanly into cards with curved corners.
    """
    from PIL import Image, ImageDraw
    scale = 2
    mask = Image.new("RGBA", (box_w * scale, box_h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, box_w * scale, box_h * scale), radius=radius * scale, fill=(255, 255, 255, 255))
    mask = mask.resize((box_w, box_h), Image.Resampling.LANCZOS)
    mask.save(out_path)
    return out_path


def _caption_bottom_y(video_height: int) -> int:
    """
    Y coordinate of the top edge of a caption placed near the bottom.
    
    The margin is a ratio of the frame height so the caption keeps the same
    visual distance from the bottom edge for 1080p as well as 1920p output.
    """
    margin = int(video_height * CAPTION_BOTTOM_MARGIN_RATIO)
    return max(0, video_height - margin)


def _caption_baseline_y(video_height: int, desc_space: int = 28) -> int:
    """
    Y coordinate of the caption baseline (where the letters sit), calibrated so that
    the bottom of the caption text + descenders + stroke + glow sits with a 30px margin
    from the bottom edge of the video frame as requested.
    """
    margin = int(round(CAPTION_BOTTOM_MARGIN_PX * (video_height / 1080.0))) if video_height > 1080 else CAPTION_BOTTOM_MARGIN_PX
    return max(0, video_height - margin - desc_space)


def _detect_line_y_position(
    video_clip: Optional[Any],
    line_start: float,
    line_end: float,
    caption_height: int,
    caption_width: int,
    video_height: int,
    video_width: int,
    position_mode: str = CAPTION_POSITION,
) -> int:
    """
    Intelligently calculate the optimal Y coordinate for a caption line.
    
    Modes:
    - 'auto': Video-aware placement. Analyzes video frame content to avoid covering
              faces/speakers and prevents colliding with existing bottom subtitles/graphics.
    - 'bottom': Fixed near bottom with 30px margin.
    - 'lower_third': Standard shorts lower-third (y ~ 74%).
    - 'middle': Centered vertically (y ~ 50%).
    - 'top': Upper third (y ~ 18%).
    """
    margin_30 = CAPTION_BOTTOM_MARGIN_PX
    if video_height > 1080:
        margin_30 = int(round(CAPTION_BOTTOM_MARGIN_PX * (video_height / 1080.0)))

    bottom_30px_y = max(0, video_height - margin_30 - caption_height)
    lower_third_y = max(0, int(video_height * 0.74 - caption_height / 2))
    middle_y = max(0, int((video_height - caption_height) * 0.50))
    top_y = max(0, int(video_height * 0.18))

    mode = (position_mode or "auto").lower().strip()
    if mode == "bottom":
        return bottom_30px_y
    elif mode == "lower_third":
        return lower_third_y
    elif mode == "middle":
        return middle_y
    elif mode == "top":
        return top_y

    # Default to auto
    if video_clip is None:
        return bottom_30px_y

    try:
        import cv2

        t_sample = max(0.0, min(float(video_clip.duration) - 0.05, (line_start + line_end) / 2.0))
        frame_rgb = video_clip.get_frame(t_sample)
        h, w, _ = frame_rgb.shape

        # Convert to BGR for cv2 analysis
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect existing subtitles/text in bottom strip (y: 84% to 98%)
        bot_y0 = int(h * 0.84)
        bot_y1 = int(h * 0.98)
        bot_strip = gray[bot_y0:bot_y1, int(w * 0.08):int(w * 0.92)]
        bot_edges = cv2.Canny(bot_strip, 80, 160)
        has_bottom_text = bool(np.mean(bot_edges) > 3.0)

        # 2. Gradient energy map (visual clutter)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = grad_mag / (np.max(grad_mag) + 1e-5)

        # 3. Detect dominant face / skin region
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)

        # Vertical distribution of skin/subject
        skin_vert = np.mean(skin, axis=1) / 255.0
        skin_top = float(np.mean(skin_vert[int(h * 0.10):int(h * 0.38)]))
        skin_mid = float(np.mean(skin_vert[int(h * 0.38):int(h * 0.70)]))
        skin_bot = float(np.mean(skin_vert[int(h * 0.70):int(h * 0.95)]))

        candidates = [
            ("bottom_30px", bottom_30px_y),
            ("lower_third", lower_third_y),
            ("middle", middle_y),
            ("top", top_y),
        ]

        x0 = max(0, (w - caption_width) // 2)
        x1 = min(w, x0 + caption_width)

        scores = {}
        for name, y in candidates:
            y0 = max(0, int(y))
            y1 = min(h, y0 + caption_height)
            clutter = float(np.mean(grad_norm[y0:y1, x0:x1]))

            penalty = 0.0
            if name == "bottom_30px":
                if has_bottom_text:
                    penalty += 10.0
                if skin_bot > 0.40:
                    penalty += 0.5
                bias = -0.015
            elif name == "lower_third":
                if has_bottom_text and y1 > bot_y0:
                    penalty += 5.0
                if skin_bot > 0.35:
                    penalty += 0.4
                bias = -0.010
            elif name == "middle":
                if skin_mid > 0.45:
                    penalty += 0.8
                bias = 0.005
            elif name == "top":
                if skin_top > 0.40:
                    penalty += 0.9
                bias = 0.025

            scores[name] = clutter + penalty + bias

        best_name, best_y = min(candidates, key=lambda c: scores[c[0]])
        return best_y
    except Exception as e:
        logger.debug(f"Auto-position fallback to bottom: {e}")
        return bottom_30px_y


def _caption_baseline_offset(font, font_size: int, stroke_width: int) -> Optional[int]:
    """
    Distance from the top of a rendered caption bitmap to its baseline.

    Must match the baseline used inside ``_render_caption_bitmap`` exactly.
    """
    try:
        from PIL import ImageFont

        font_path = _resolve_pil_font(font)
        if not font_path:
            return None

        pil_font = ImageFont.truetype(font_path, font_size)
        ascent, _ = pil_font.getmetrics()
        return ascent + max(0, int(stroke_width)) + 4
    except Exception:
        return None


_HB_CACHE: Dict[str, Any] = {}


def _get_harfbuzz_fonts(font_path: str):
    """Load and cache HarfBuzz & FreeType font objects for OpenType text shaping."""
    if font_path in _HB_CACHE:
        return _HB_CACHE[font_path]
    try:
        import uharfbuzz as hb
        import freetype

        with open(font_path, "rb") as f:
            font_bytes = f.read()

        blob = hb.Blob(font_bytes)
        hface = hb.Face(blob)
        hfont = hb.Font(hface)
        fface = freetype.Face(font_path)

        _HB_CACHE[font_path] = (hb, hfont, freetype, fface)
        return _HB_CACHE[font_path]
    except Exception as e:
        logger.debug(f"HarfBuzz not available for {font_path}: {e}")
        return None


def _measure_harfbuzz_advance(word: str, font_path: str, font_size: int) -> float:
    """Accurately measure text advance using HarfBuzz OpenType shaping."""
    hb_objs = _get_harfbuzz_fonts(font_path)
    if not hb_objs:
        return len(word) * font_size * 0.65
    hb_mod, hfont, ft_mod, fface = hb_objs
    fface.set_pixel_sizes(0, font_size)
    hfont.scale = (font_size * 64, font_size * 64)
    buf = hb_mod.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()
    buf.script = "Deva"
    buf.language = "hi"
    hb_mod.shape(hfont, buf)
    return sum(p.x_advance for p in buf.glyph_positions) / 64.0


def _shape_harfbuzz_word(
    word: str,
    font_path: str,
    font_size: int,
    stroke_width: int,
    text_color,
    stroke_color,
    shadow: bool = False,
):
    """
    Render a single word with full OpenType complex text shaping (HarfBuzz).
    Ensures 100% correct Devanagari ligatures (conjuncts like धर्म, दुष्कर्म, अहिंसा, निंदा)
    and proper matra placement.
    """
    hb_objs = _get_harfbuzz_fonts(font_path)
    if not hb_objs:
        return None, None, None

    from PIL import Image, ImageFilter
    import numpy as np

    hb_mod, hfont, ft_mod, fface = hb_objs
    fface.set_pixel_sizes(0, font_size)
    hfont.scale = (font_size * 64, font_size * 64)

    buf = hb_mod.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()
    buf.script = "Deva"
    buf.language = "hi"
    hb_mod.shape(hfont, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    x_cursor = 0
    glyph_runs = []
    min_x, max_x = 0, 0
    min_y, max_y = 0, 0

    for info, pos in zip(infos, positions):
        gid = info.codepoint
        fface.load_glyph(gid, ft_mod.FT_LOAD_RENDER | ft_mod.FT_LOAD_TARGET_NORMAL)
        slot = fface.glyph
        bitmap = slot.bitmap

        gx = (x_cursor + pos.x_offset) / 64.0 + slot.bitmap_left
        gy = -(pos.y_offset / 64.0 + slot.bitmap_top)

        bw, bh = bitmap.width, bitmap.rows
        if bw > 0 and bh > 0:
            arr = np.array(bitmap.buffer, dtype=np.uint8).reshape((bh, bw))
            glyph_runs.append((gx, gy, bw, bh, arr))
            min_x = min(min_x, gx)
            max_x = max(max_x, gx + bw)
            min_y = min(min_y, gy)
            max_y = max(max_y, gy + bh)

        x_cursor += pos.x_advance

    total_advance = x_cursor / 64.0
    pad = stroke_width * 2 + 10

    width = int(max(total_advance, max_x) - min_x + 2 * pad) + 4
    height = int(max_y - min_y + 2 * pad) + 4

    alpha_img = Image.new("L", (width, height), 0)
    base_x = pad - min_x
    base_y = pad - min_y

    for gx, gy, bw, bh, arr in glyph_runs:
        px = int(round(base_x + gx))
        py = int(round(base_y + gy))
        g_img = Image.fromarray(arr)
        temp = Image.new("L", (width, height), 0)
        temp.paste(g_img, (px, py))
        alpha_img = Image.fromarray(np.maximum(np.array(alpha_img), np.array(temp)))

    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    if shadow:
        sh_mask = alpha_img.filter(ImageFilter.GaussianBlur(radius=2))
        sh_layer = Image.new("RGBA", (width, height), (0, 0, 0, 180))
        sh_layer.putalpha(sh_mask)
        out.paste(sh_layer, (2, 3), sh_layer)

    if stroke_width > 0:
        stroke_mask = alpha_img.filter(ImageFilter.MaxFilter(stroke_width * 2 + 1))
        stroke_layer = Image.new("RGBA", (width, height), stroke_color)
        stroke_layer.putalpha(stroke_mask)
        out = Image.alpha_composite(out, stroke_layer)

    fill_layer = Image.new("RGBA", (width, height), text_color)
    fill_layer.putalpha(alpha_img)
    out = Image.alpha_composite(out, fill_layer)

    bbox = out.getchannel("A").getbbox()
    if bbox:
        cropped = out.crop((bbox[0], 0, bbox[2], height))
        return cropped, base_y, total_advance
    return out, base_y, total_advance


def _create_card_footer_clip(text: str, font_path: str, font_size: int, duration: float) -> Any:
    """
    Render card footer text (e.g. viral title / hook line) with HarfBuzz complex shaping & emoji support.
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from moviepy import ImageClip

    # Normalize Hindi spelling in footer text
    words = text.split()
    corrected_words = [_correct_hindi_word(w) for w in words]
    text = " ".join(corrected_words)

    emoji_candidates = [
        r"C:\Windows\Fonts\seguiemj.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/ancient-scripts/Symbola.ttf",
    ]
    emoji_font_path = next((p for p in emoji_candidates if os.path.exists(p)), None)
    has_emoji_font = emoji_font_path is not None

    base_font = ImageFont.truetype(font_path, font_size)
    emoji_font = ImageFont.truetype(emoji_font_path, font_size) if has_emoji_font else base_font

    runs = []
    cur_text = ""
    cur_is_emoji = False
    for ch in text:
        is_em = (ord(ch) > 0x2000 and not (0x0900 <= ord(ch) <= 0x097F))
        if cur_text == "":
            cur_text = ch
            cur_is_emoji = is_em
        elif is_em == cur_is_emoji:
            cur_text += ch
        else:
            runs.append((cur_text, cur_is_emoji))
            cur_text = ch
            cur_is_emoji = is_em
    if cur_text:
        runs.append((cur_text, cur_is_emoji))

    rendered_runs = []
    for t, is_em in runs:
        if is_em and has_emoji_font:
            w = emoji_font.getlength(t)
            rendered_runs.append((t, is_em, None, w))
        else:
            w_img, b_y, adv = _shape_harfbuzz_word(
                word=t,
                font_path=font_path,
                font_size=font_size,
                stroke_width=0,
                text_color=_rgba(CARD_FOOTER_COLOR),
                stroke_color=(0, 0, 0, 0),
            )
            if w_img is not None:
                rendered_runs.append((t, is_em, w_img, adv))
            else:
                w = base_font.getlength(t)
                rendered_runs.append((t, is_em, None, w))

    total_w = sum(adv for _, _, _, adv in rendered_runs)
    foot_im = Image.new("RGBA", (int(total_w) + 30, int(font_size * 2)), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(foot_im)
    cx = 15
    cy = int(font_size * 0.35)
    for t, is_em, w_img, adv in rendered_runs:
        if is_em and has_emoji_font:
            fdraw.text((cx, cy), t, font=emoji_font, fill=(255, 255, 255, 255), embedded_color=True)
        elif w_img is not None:
            foot_im.alpha_composite(w_img, dest=(int(cx), 0))
        else:
            fdraw.text((cx, cy), t, font=base_font, fill=_rgba(CARD_FOOTER_COLOR))
        cx += adv

    foot_clip = ImageClip(np.array(foot_im), transparent=True, duration=duration)
    foot_clip = foot_clip.with_position(("center", CARD_BOX_Y + CARD_BOX_HEIGHT + 30))
    return foot_clip


def _measure_line_width(texts: List[str], font: str, font_size: int) -> float:
    """Measure the total line width of a list of words using HarfBuzz OpenType advance."""
    try:
        font_path = _resolve_pil_font(font)
        if font_path and os.path.isfile(font_path):
            sw = max(1, int(CAPTION_GROUP_STROKE_WIDTH))
            pad_x = sw + 4
            spacing = max(8, int(font_size * CAPTION_WORD_SPACING))
            active_size = int(round(font_size * 1.38))
            slot_widths = [
                max(
                    _measure_harfbuzz_advance(t, font_path, font_size) + 2 * pad_x,
                    _measure_harfbuzz_advance(t, font_path, active_size) + 2 * pad_x,
                )
                for t in texts
            ]
            return sum(slot_widths) + spacing * max(0, len(texts) - 1)
    except Exception:
        pass
    return sum(len(t) for t in texts) * font_size * 0.65


def _chunk_words_into_lines(
    words: List[Dict],
    font: str,
    max_usable_width: int,
    base_size: int = CAPTION_FONT_SIZE,
    max_pause: float = 0.55,
) -> List[List[Dict]]:
    """
    Auto-fill words dynamically into natural speech lines based on screen width.
    
    Rather than constraining to a fixed word count (e.g. 4 words), words auto-fill
    into each line as long as they fit within ``max_usable_width``.  A line break occurs
    naturally on:
    1. Speech pauses (> 0.55s silence) between words
    2. Strong sentence-ending punctuation (., !, ?, |, ।, —)
    3. When adding the next word would exceed the line width
    
    Also balances phrases so we avoid leaving awkward 1-word orphan lines before a pause.
    """
    lines: List[List[Dict]] = []
    current: List[Dict] = []
    n = len(words)
    i = 0

    while i < n:
        w = words[i]
        if not current:
            current.append(w)
            i += 1
            continue

        pause = float(w.get("start", 0)) - float(current[-1].get("end", 0))
        prev_word = current[-1].get("text", "")
        has_punct = any(p in prev_word for p in [".", ",", "!", "?", "|", "।", "—"])

        # Natural speech pause or punctuation ends the line
        if pause > max_pause or has_punct:
            lines.append(current)
            current = [w]
            i += 1
            continue

        cand_texts = [x["text"] for x in current] + [w["text"]]
        cand_w = _measure_line_width(cand_texts, font, base_size)

        if cand_w <= max_usable_width:
            current.append(w)
            i += 1
        elif cand_w <= max_usable_width * 1.08:
            # Fits comfortably with a tiny 2-3px font auto-scaling
            current.append(w)
            i += 1
        else:
            # Line is full. Check remaining words in this speech clause before next pause.
            rem = 1
            for j in range(i + 1, min(i + 4, n)):
                if (float(words[j].get("start", 0)) - float(words[j - 1].get("end", 0))) > max_pause:
                    break
                rem += 1

            # If only 1 word remains in this clause and current has >= 4 words,
            # balance by shifting 1-2 words so we don't leave a lonely 1-word line.
            if rem == 1 and len(current) >= 4:
                shift_count = 2 if len(current) >= 5 else 1
                moved = current[-shift_count:]
                current = current[:-shift_count]
                lines.append(current)
                current = moved + [w]
                i += 1
            else:
                lines.append(current)
                current = [w]
                i += 1

    if current:
        lines.append(current)
    return lines


def _render_caption_line_bitmap(
    texts: List[str],
    active_index: int,
    font,
    font_size: int,
    active_color: str = CAPTION_COLOR,
    inactive_color: str = CAPTION_INACTIVE_COLOR,
    stroke_color: str = CAPTION_STROKE_COLOR,
    stroke_width: int = CAPTION_GROUP_STROKE_WIDTH,
    active_scale: float = CAPTION_ACTIVE_SCALE,
    glow_color: str = CAPTION_GLOW_COLOR,
    glow_radius_ratio: float = CAPTION_GLOW_RADIUS,
    max_width: Optional[int] = None,
    pop_phase: str = "sustain",
) -> Tuple[Optional[Any], Optional[int]]:
    """
    Render a line of words where ONLY the word at ``active_index`` is highlighted with
    modern punchy pop animation, 3D text shadow, and multi-layer neon glow aura.
    
    - The active word is rendered bigger and in vibrant yellow with a warm glow behind it.
    - All other words in the line are rendered normal size in crisp white.
    - Each word sits in a fixed-width slot so words never jitter or shift
      horizontally when the highlight moves from word to word.
    - All words share the exact same baseline across every active state.
    
    Args:
        texts: List of words in the line
        active_index: Which word is being spoken right now
        font: Font path or name
        font_size: Base font size for normal words
        active_color: Fill colour for the active word
        inactive_color: Fill colour for normal words
        stroke_color: Outline colour
        stroke_width: Outline thickness
        active_scale: Size multiplier for active word
        glow_color: Glow halo colour
        glow_radius_ratio: Glow blur radius ratio
        max_width: Maximum allowed width in pixels (safe margin)
        pop_phase: "pop" (punch-in bounce) or "sustain" (steady highlight)
        
    Returns:
        ``(image, baseline_offset)`` tuple, or ``(None, None)`` on failure.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except Exception as e:
        logger.warning(f"PIL is required for captions: {e}")
        return None, None

    if not texts:
        return None, None

    font_path = _resolve_pil_font(font)
    if not font_path:
        return None, None

    sw = max(1, int(stroke_width))
    if pop_phase == "pop":
        cur_active_scale = 1.38
        glow_alpha = 0.95
        glow_radius_mult = 0.26
        y_lift = 3
    else:
        cur_active_scale = 1.25
        glow_alpha = 0.70
        glow_radius_mult = 0.20
        y_lift = 0

    normal_size = font_size
    active_size = int(round(font_size * cur_active_scale))
    max_active_size = int(round(font_size * 1.38))

    pad_x = sw + 4
    slot_widths = []
    for t in texts:
        nw = _measure_harfbuzz_advance(t, font_path, normal_size) + 2 * pad_x
        maw = _measure_harfbuzz_advance(t, font_path, max_active_size) + 2 * pad_x
        slot_widths.append(max(nw, maw))

    spacing = max(8, int(font_size * CAPTION_WORD_SPACING))
    total_line_w = sum(slot_widths) + spacing * max(0, len(texts) - 1)

    glow_radius = max(3, int(font_size * glow_radius_mult))
    glow_spread = int(glow_radius * 1.5)
    canvas_pad = glow_radius * 3 + 12

    # Render each word with HarfBuzz OpenType complex text shaping
    rendered_words = []
    for i, t in enumerate(texts):
        is_active = (i == active_index)
        sz = active_size if is_active else normal_size
        col = _rgba(active_color) if is_active else _rgba(inactive_color)
        w_img, base_y, _ = _shape_harfbuzz_word(
            word=t,
            font_path=font_path,
            font_size=sz,
            stroke_width=sw + (1 if is_active else 0),
            text_color=col,
            stroke_color=_rgba(stroke_color),
            shadow=is_active,
        )
        if w_img is None:
            return None, None
        rendered_words.append((w_img, base_y, is_active))

    max_ascent_space = max(base_y for _, base_y, _ in rendered_words)
    max_descent_space = max(w_img.height - base_y for w_img, base_y, _ in rendered_words)

    fixed_baseline = canvas_pad + max_ascent_space + 4
    canvas_w = int(total_line_w + 2 * canvas_pad)
    canvas_h = int(fixed_baseline + max_descent_space + canvas_pad + 8)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    glow_mask = Image.new("L", (canvas_w, canvas_h), 0)

    x = canvas_pad
    for i, (w_img, base_y, is_active) in enumerate(rendered_words):
        slot_w = slot_widths[i]
        word_x = int(x + (slot_w - w_img.width) / 2)
        word_y = int(fixed_baseline - base_y - (y_lift if is_active else 0))

        canvas.alpha_composite(w_img, dest=(word_x, word_y))
        if is_active:
            glow_mask.paste(w_img.getchannel("A"), (word_x, word_y))

        x += slot_w + spacing

    # Multi-layer Neon Glow
    glow1 = Image.new("RGBA", (canvas_w, canvas_h), (255, 230, 0, 255))
    mask1 = glow_mask.point(lambda v: int(v * glow_alpha))
    glow1.putalpha(mask1)
    glow1 = glow1.filter(ImageFilter.GaussianBlur(radius=glow_radius // 2))

    glow2 = Image.new("RGBA", (canvas_w, canvas_h), (255, 160, 0, 200))
    mask2 = glow_mask.point(lambda v: int(v * (glow_alpha * 0.70)))
    glow2.putalpha(mask2)
    glow2 = glow2.filter(ImageFilter.GaussianBlur(radius=glow_radius))

    neon = Image.alpha_composite(glow2, glow1)
    final = Image.alpha_composite(neon, canvas)

    # Tight symmetrical crop
    line_center = canvas_w / 2
    half_line_w = total_line_w / 2 + glow_spread + 4
    crop_x0 = max(0, int(line_center - half_line_w))
    crop_x1 = min(canvas_w, int(line_center + half_line_w))
    crop_y0 = max(0, int(fixed_baseline - max_ascent_space - glow_spread - 4))
    crop_y1 = min(canvas_h, int(fixed_baseline + max_descent_space + glow_spread + 4))

    cropped = final.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    baseline_offset = fixed_baseline - crop_y0

    # Ensure line NEVER exceeds max_width (safe margins on left and right)
    if max_width and cropped.width > max_width:
        ratio = max_width / cropped.width
        new_size = (max_width, max(1, int(cropped.height * ratio)))
        cropped = cropped.resize(new_size, Image.LANCZOS)
        baseline_offset = max(1, int(round(baseline_offset * ratio)))

    return cropped, baseline_offset


def _fit_group_font_size(
    texts: List[str],
    font,
    max_usable_width: int,
    base_size: int = CAPTION_FONT_SIZE,
) -> int:
    """
    Find the font size that keeps the entire line comfortably inside ``max_usable_width``.
    """
    if not texts:
        return base_size

    font_path = _resolve_pil_font(font)
    if font_path and os.path.isfile(font_path):
        try:
            sw = max(1, int(CAPTION_GROUP_STROKE_WIDTH))
            pad_x = sw + 4
            spacing_ratio = CAPTION_WORD_SPACING

            active_size = int(round(base_size * 1.38))
            slot_widths = [
                max(
                    _measure_harfbuzz_advance(t, font_path, base_size) + 2 * pad_x,
                    _measure_harfbuzz_advance(t, font_path, active_size) + 2 * pad_x,
                )
                for t in texts
            ]
            spacing = int(base_size * spacing_ratio)
            total_w = sum(slot_widths) + spacing * max(0, len(texts) - 1)

            if total_w > max_usable_width:
                scale = max_usable_width / total_w
                return max(20, min(base_size, int(base_size * scale)))
            return base_size
        except Exception:
            pass

    return base_size


def _create_caption_clips(
    words: List[Dict],
    video_duration: float,
    font: str,
    video_height: int = 1920,
    video_width: int = 1080,
    cancel_token: Optional[CancellationToken] = None,
    video_clip: Optional[Any] = None,
    caption_position: str = CAPTION_POSITION,
    layout: str = DEFAULT_VIDEO_LAYOUT,
) -> List[Any]:
    """
    Create animated karaoke caption clips for a video.
    
    Words are chunked into lines of up to :data:`CAPTION_GROUP_SIZE` words (at least
    4 words when continuous speech allows).  Each line stays stationary on screen
    while the spoken word lights up in vibrant yellow, grows ~30% bigger, and glows.
    Once spoken, it returns to normal white and the next word lights up.
    
    Args:
        words: List of word dicts with relative timestamps
        video_duration: Total duration of the clipped video
        font: Font to use for text
        video_height: Height of the video (for safe lower-third positioning)
        video_width: Width of the video (to enforce side safe margins)
        cancel_token: Optional cancellation token
        video_clip: Optional MoviePy video clip for visual frame analysis
        caption_position: Positioning mode ('auto', 'bottom', 'lower_third', 'middle', 'top')
        layout: Presentation layout ('card' for black bg + rounded box, or 'full' for full crop)
        
    Returns:
        List of MoviePy ImageClip objects
    """
    text_clips = []

    words = sorted(
        words,
        key=lambda w: float(w.get("start", 0) or 0),
    )

    is_card = (layout == "card")
    base_font_size = CARD_CAPTION_FONT_SIZE if is_card else CAPTION_FONT_SIZE
    usable_width = int(video_width * 0.90) if is_card else max(80, int(video_width * CAPTION_MAX_WIDTH_RATIO))

    # Clean words
    clean_words = []
    for word_data in words:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        raw_word = word_data.get("word", "")
        sanitized = _sanitize_caption_text(raw_word)
        if not sanitized:
            continue

        start = float(word_data.get("start", 0) or 0)
        end = float(word_data.get("end", start) or start)
        if start >= video_duration:
            continue
        end = min(end, video_duration)
        if end <= start:
            continue

        clean_words.append({"text": sanitized, "start": start, "end": end})

    if not clean_words:
        logger.info("Created 0 text clips")
        return text_clips

    # Auto-fill words dynamically into natural lines based on screen width
    lines = _chunk_words_into_lines(clean_words, font, usable_width, base_font_size)
    logger.info(f"Auto-filled {len(clean_words)} words into {len(lines)} dynamic caption lines")

    # Space below baseline for letters + descender + stroke + glow
    active_desc = int(round(base_font_size * CAPTION_ACTIVE_SCALE * 0.28))
    desc_space = max(26, active_desc + CAPTION_GROUP_STROKE_WIDTH + 10)
    baseline_y = _caption_baseline_y(video_height, desc_space)

    bottom_margin = int(round(CAPTION_BOTTOM_MARGIN_PX * (video_height / 1080.0))) if video_height > 1080 else CAPTION_BOTTOM_MARGIN_PX

    for line_idx, line in enumerate(lines):
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        texts = [w["text"] for w in line]
        font_size = _fit_group_font_size(texts, font, usable_width, base_font_size)

        # Detect optimal Y position for this line
        line_start = line[0]["start"]
        line_end = line[-1]["end"]
        estimated_h = int(font_size * 2.2)

        if is_card:
            pos_key = (caption_position or "").lower().strip()
            if pos_key == "bottom":
                line_y = CARD_BOX_Y + CARD_BOX_HEIGHT - CARD_CAPTION_MARGIN - estimated_h
            elif pos_key == "middle":
                line_y = CARD_BOX_Y + (CARD_BOX_HEIGHT - estimated_h) // 2
            else:
                # Default card layout: 20px margin above the video box in top black area
                line_y = max(20, CARD_BOX_Y - CARD_CAPTION_MARGIN - estimated_h)
        else:
            line_y = _detect_line_y_position(
                video_clip=video_clip,
                line_start=line_start,
                line_end=line_end,
                caption_height=estimated_h,
                caption_width=usable_width,
                video_height=video_height,
                video_width=video_width,
                position_mode=caption_position,
            )

        # Precompute next line start for clean hold calculation
        next_line_start = lines[line_idx + 1][0]["start"] if (line_idx + 1 < len(lines)) else video_duration

        for active_idx in range(len(line)):
            active_word = line[active_idx]
            start = active_word["start"]

            # Duration for this word's active state:
            if active_idx + 1 < len(line):
                # Word state ends when the NEXT word in this line starts
                end = min(line[active_idx + 1]["start"], video_duration)
            else:
                # Last word of the line stays until next line starts (or small hold)
                hold_end = active_word["end"] + min(0.6, CAPTION_MAX_HOLD)
                end = min(next_line_start, hold_end, video_duration)

            if end <= start:
                end = min(video_duration, start + 0.08)
            duration = end - start

            try:
                # 1. Render sustain bitmap to determine anchor baseline
                bitmap_sus, offset_sus = _render_caption_line_bitmap(
                    texts=texts,
                    active_index=active_idx,
                    font=font,
                    font_size=font_size,
                    active_color=CAPTION_COLOR,
                    inactive_color=CAPTION_INACTIVE_COLOR,
                    stroke_color=CAPTION_STROKE_COLOR,
                    stroke_width=CAPTION_GROUP_STROKE_WIDTH,
                    active_scale=CAPTION_ACTIVE_SCALE,
                    glow_color=CAPTION_GLOW_COLOR,
                    glow_radius_ratio=CAPTION_GLOW_RADIUS,
                    max_width=usable_width,
                    pop_phase="sustain",
                )
                if bitmap_sus is None or offset_sus is None:
                    continue

                # Target baseline Y coordinate (keeps text baseline rock-solid)
                if is_card:
                    pos_key = (caption_position or "").lower().strip()
                    if pos_key == "bottom":
                        target_baseline_y = (CARD_BOX_Y + CARD_BOX_HEIGHT - CARD_CAPTION_MARGIN) - (bitmap_sus.height - offset_sus)
                    elif pos_key == "middle":
                        target_baseline_y = (CARD_BOX_Y + CARD_BOX_HEIGHT // 2) + offset_sus - bitmap_sus.height // 2
                    else:
                        # Exactly 20px margin above clip box
                        target_baseline_y = (CARD_BOX_Y - CARD_CAPTION_MARGIN) - (bitmap_sus.height - offset_sus)
                elif (caption_position or "").lower().strip() == "bottom":
                    target_baseline_y = (video_height - bottom_margin) - (bitmap_sus.height - offset_sus)
                else:
                    target_baseline_y = min((video_height - bottom_margin) - (bitmap_sus.height - offset_sus), line_y + offset_sus)

                POP_DURATION = 0.09  # 90ms punch-in bounce
                pop_t = min(POP_DURATION, max(0.04, duration * 0.35))

                if duration > 0.08:
                    # Phase 1: Pop Bounce (punchy scale & intense neon burst)
                    bitmap_pop, offset_pop = _render_caption_line_bitmap(
                        texts=texts,
                        active_index=active_idx,
                        font=font,
                        font_size=font_size,
                        active_color=CAPTION_COLOR,
                        inactive_color=CAPTION_INACTIVE_COLOR,
                        stroke_color=CAPTION_STROKE_COLOR,
                        stroke_width=CAPTION_GROUP_STROKE_WIDTH,
                        active_scale=CAPTION_ACTIVE_SCALE,
                        glow_color=CAPTION_GLOW_COLOR,
                        glow_radius_ratio=CAPTION_GLOW_RADIUS,
                        max_width=usable_width,
                        pop_phase="pop",
                    )
                    if bitmap_pop is not None and offset_pop is not None:
                        pop_clip = ImageClip(np.array(bitmap_pop), transparent=True, duration=pop_t)
                        pop_y = int(target_baseline_y - offset_pop)
                        pop_clip = pop_clip.with_duration(pop_t)
                        pop_clip = pop_clip.with_position(("center", pop_y))
                        pop_clip = pop_clip.with_start(start)
                        pop_clip = pop_clip.with_end(start + pop_t)
                        text_clips.append(pop_clip)

                    # Phase 2: Settle Sustain
                    sus_dur = duration - pop_t
                    sus_clip = ImageClip(np.array(bitmap_sus), transparent=True, duration=sus_dur)
                    sus_y = int(target_baseline_y - offset_sus)
                    sus_clip = sus_clip.with_duration(sus_dur)
                    sus_clip = sus_clip.with_position(("center", sus_y))
                    sus_clip = sus_clip.with_start(start + pop_t)
                    sus_clip = sus_clip.with_end(end)
                    text_clips.append(sus_clip)
                else:
                    # Short duration: single sustain clip
                    text_clip = ImageClip(np.array(bitmap_sus), transparent=True, duration=duration)
                    sus_y = int(target_baseline_y - offset_sus)
                    text_clip = text_clip.with_duration(duration)
                    text_clip = text_clip.with_position(("center", sus_y))
                    text_clip = text_clip.with_start(start)
                    text_clip = text_clip.with_end(end)
                    text_clips.append(text_clip)

            except Exception as e:
                logger.warning(f"Failed to create caption clip for {texts} active={active_idx}: {e}")
                continue

    logger.info(
        f"Created {len(text_clips)} caption clips "
        f"({len(lines)} lines, active word karaoke glowing)"
    )
    return text_clips


def _create_word_clips(
    words: List[Dict],
    video_duration: float,
    font: str,
    video_height: int = 1920,
    video_width: int = 1080,
    cancel_token: Optional[CancellationToken] = None,
    video_clip: Optional[Any] = None,
    caption_position: str = CAPTION_POSITION,
    layout: str = DEFAULT_VIDEO_LAYOUT,
) -> List[Any]:
    """
    Backwards-compatible alias for :func:`_create_caption_clips`.
    """
    return _create_caption_clips(
        words=words,
        video_duration=video_duration,
        font=font,
        video_height=video_height,
        video_width=video_width,
        cancel_token=cancel_token,
        video_clip=video_clip,
        caption_position=caption_position,
        layout=layout,
    )


def _format_ass_time(sec: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cs"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _generate_ass_subtitles(
    words: List[Dict],
    video_duration: float,
    layout: str = "card",
    caption_position: str = "below",
    footer_title: Optional[str] = None,
    output_ass_path: str = "captions.ass",
    card_top_y: Optional[int] = None,
    card_bottom_y: Optional[int] = None,
) -> str:
    """
    Generate Advanced SubStation Alpha (.ass) subtitles with Hormozi-style animated karaoke.
    Active word pops slightly and lights up in vibrant yellow, inactive words remain in crisp white.
    Executed natively by FFmpeg libass at 100+ fps (30-50x faster than MoviePy).
    """
    clean_words = []
    for w in sorted(words, key=lambda x: float(x.get("start", 0))):
        text = _sanitize_caption_text(w.get("word", "")).strip()
        if not text:
            continue
        st = max(0.0, float(w.get("start", 0)))
        en = min(video_duration, float(w.get("end", st)))
        # Guard against Whisper runaway duration glitch (clamps single word duration)
        max_duration = max(0.35, min(1.2, len(text) * 0.12))
        if en <= st or (en - st) > max_duration:
            en = st + max_duration
        clean_words.append({"text": text, "start": st, "end": en})

    # Chunk into natural lines of 4-6 words or on pause/punctuation
    lines = []
    curr = []
    for w in clean_words:
        if curr:
            pause = w["start"] - curr[-1]["end"]
            has_punct = any(p in curr[-1]["text"] for p in [".", "?", "!", "।", "—"])
            # Break immediately on sentence end, 5 words, or speech pause > 0.60s
            if len(curr) >= 5 or has_punct or pause > 0.60:
                lines.append(curr)
                curr = []
        curr.append(w)
    if curr:
        lines.append(curr)

    is_card = (layout == "card")
    pos_key = (caption_position or "").lower().strip()
    
    # Calculate vertical position with safety margins so captions NEVER overlap the video card
    if is_card:
        top_bound = card_top_y if card_top_y is not None else 673
        bottom_bound = card_bottom_y if card_bottom_y is not None else 1247
        
        if pos_key in ("above", "top"):
            # Option: Connected tightly just above the video card (~80px above card)
            pos_y = max(260, top_bound - 80)
        elif pos_key in ("inside", "inside_video"):
            # Option: Inside the video near the bottom edge
            pos_y = bottom_bound - 65
        elif pos_key == "middle":
            # Option: Center of the video box
            pos_y = (top_bound + bottom_bound) // 2
        else:
            # Default ("below", "bottom", "auto"): Right below the video card (~95px under the card)
            # Very natural on mobile, doesn't block video or text!
            pos_y = bottom_bound + 95
        
        # Keep footer nicely separated from captions
        if pos_y > bottom_bound:
            footer_y = min(1780, pos_y + 150)
        else:
            footer_y = max(1480, min(1750, bottom_bound + 250))
    else:
        if pos_key == "top":
            pos_y = 380
        elif pos_key == "middle":
            pos_y = 960
        else:
            # Default Full Shorts: Lower third
            pos_y = 1420
        footer_y = 1800

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Karaoke,Noto Sans Devanagari,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,8,40,40,40,1",
        "Style: Footer,Noto Sans Devanagari,34,&H00D0D0D0,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    # Footer title for Card Style
    if is_card and footer_title:
        clean_footer = footer_title.split("|")[0].split("!")[0].strip()
        if len(clean_footer) > 38:
            clean_footer = clean_footer[:38] + "..."
        clean_footer = clean_footer.replace("{", "(").replace("}", ")")
        ass_lines.append(
            f"Dialogue: 0,0:00:00.00,{_format_ass_time(video_duration)},Footer,,0,0,0,,{{\\pos(540,{footer_y})}}{clean_footer}"
        )

    # Karaoke events
    for line_idx, line in enumerate(lines):
        next_line_start = lines[line_idx + 1][0]["start"] if (line_idx + 1 < len(lines)) else video_duration
        line_natural_end = line[-1]["end"] + 0.40
        for active_idx, active_w in enumerate(line):
            st = active_w["start"]
            if active_idx + 1 < len(line):
                en = line[active_idx + 1]["start"]
            else:
                # Disappear within 0.4s after the last word finishes, or max 2.5s hold, NEVER stay stuck!
                en = min(next_line_start, line_natural_end, st + 2.5, video_duration)
            if en <= st:
                en = st + 0.10

            parts = []
            for i, item in enumerate(line):
                safe_text = item["text"].replace("{", "(").replace("}", ")")
                if i == active_idx:
                    parts.append(f"{{\\c&H00D7FF&\\fscx120\\fscy120}}{safe_text}{{\\r\\c&HFFFFFF&}}")
                else:
                    parts.append(safe_text)
            line_str = " ".join(parts)
            ass_lines.append(
                f"Dialogue: 0,{_format_ass_time(st)},{_format_ass_time(en)},Karaoke,,0,0,0,,{{\\pos(540,{pos_y})}}{line_str}"
            )

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_lines))

    return output_ass_path


def create_short(
    video_path: str,
    transcript_words: List[Dict],
    clip_data: Dict[str, Any],
    output_path: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
    transcript_full_text: Optional[str] = None,
    caption_position: str = CAPTION_POSITION,
    layout: str = DEFAULT_VIDEO_LAYOUT,
) -> str:
    """
    Create a YouTube Short from the video clip with animated captions.
    
    This is the main Phase 3 function that:
    1. Trims the video to the exact clip timestamps
    2. Crop-s to 9:16 aspect ratio
    3. Generates word-by-word animated captions (Hormozi-style)
    4. Composites and exports the final Short
    
    Args:
        video_path: Path to the downloaded video from Phase 1
        transcript_words: Word-level transcript segments from Phase 1
        clip_data: Clip recommendation from Phase 2
            - clip_start_time: Start time in seconds
            - clip_end_time: End time in seconds
            - clip_text: Text content of the clip
        output_path: Optional custom output path
        cancel_token: Optional token used to stop FFmpeg/MoviePy immediately
        transcript_full_text: Optional complete transcript text.  Used to
            rebuild word timings when the transcript only carries segment
            level timestamps.
    
    Returns:
        Path to the final exported short video
    
    Raises:
        VideoProcessingError: If video processing fails
        CaptionGenerationError: If caption generation fails
    """
    # Ensure output directory exists
    _ensure_output_dir()

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    # Repair any missing word-level timings up-front so the captions can
    # never end up empty (older transcripts / segment-only back-ends).
    ensure_word_timestamps(transcript_words, transcript_full_text)
    
    # Extract clip parameters
    clip_start = clip_data.get("clip_start_time", 0)
    clip_end = clip_data.get("clip_end_time", 30)

    # Ensure complete narrative sense & sentence closure if not already refined
    if transcript_words and not clip_data.get("is_boundary_refined"):
        try:
            from phase2_brain import refine_clip_boundaries
            t_data = {"segments": transcript_words, "full_text": transcript_full_text or ""}
            r_start, r_end, r_text = refine_clip_boundaries(t_data, clip_start, clip_end)
            if r_end > r_start:
                logger.info(
                    f"Refined ending boundary in Phase 3 for complete sentence closure: "
                    f"{clip_start:.2f}s-{clip_end:.2f}s -> {r_start:.2f}s-{r_end:.2f}s"
                )
                clip_start, clip_end = r_start, r_end
                clip_data["clip_start_time"] = clip_start
                clip_data["clip_end_time"] = clip_end
                clip_data["is_boundary_refined"] = True
                if r_text:
                    clip_data["clip_text"] = r_text
        except Exception as e:
            logger.debug(f"Phase 3 boundary closure check skipped: {e}")
    
    if clip_end <= clip_start:
        raise VideoProcessingError("Invalid clip times: end must be after start")
    
    clip_duration = clip_end - clip_start
    logger.info(f"Creating Short from {clip_start:.2f}s to {clip_end:.2f}s (duration: {clip_duration:.2f}s)")
    
    if not (30 <= clip_duration <= 60):
        logger.warning(f"Clip duration {clip_duration}s is outside recommended 30-60s range")
    
    # Set output path
    if output_path is None:
        output_path = str(_ensure_output_dir() / DEFAULT_OUTPUT_FILENAME)
    else:
        # Make sure the target folder exists, otherwise the final
        # os.replace() below fails with "path not found".
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        trimmed_path = str(temp_dir_path / "trimmed.mp4")
        cropped_path = str(temp_dir_path / "cropped.mp4")
        final_path = str(temp_dir_path / "final_composite.mp4")

        video_clip = None
        final_clip = None
        text_clips = []

        # Filter and adjust transcript for clip upfront
        filtered_words = _filter_words_for_clip(
            transcript_words,
            clip_start,
            clip_end,
            transcript_full_text,
        )

        # Probe source video stream dimensions upfront to calculate safe geometry
        logger.info("\n--- STEP 1: Probing source video stream ---")
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream:
            raise VideoProcessingError("Could not find video stream in source file")

        width = int(video_stream['width'])
        height = int(video_stream['height'])
        has_audio = any(s for s in probe['streams'] if s['codec_type'] == 'audio')
        logger.info(f"Source video: {width}x{height}, Audio stream: {has_audio}")

        # Compute dynamic Card dimensions that preserve 100% of video content (No-Crop!)
        # Fits horizontal or vertical videos cleanly without cutting off side text or presentations
        aspect = width / height if height > 0 else 1.777
        if aspect >= 1.0:
            # Landscape or square: Fit width with 30px side margins on 1080 canvas
            card_w = 1020
            card_h = int(card_w / aspect)
            if card_h > 1200:
                card_h = 1200
                card_w = int(card_h * aspect)
        else:
            # Vertical video (e.g. 9:16)
            card_h = 1600
            card_w = int(card_h * aspect)
            if card_w > 1020:
                card_w = 1020
                card_h = int(card_w / aspect)

        card_w = card_w - (card_w % 2)
        card_h = card_h - (card_h % 2)
        card_x = (CANVAS_WIDTH - card_w) // 2
        card_y = (CANVAS_HEIGHT - card_h) // 2
        card_bottom_y = card_y + card_h

        ass_filename = f"captions_{int(time.time())}_{os.getpid()}.ass"
        ass_path = str(OUTPUT_DIR / ass_filename)
        ass_rel_path = f"output/{ass_filename}"
        ass_generated = False

        if filtered_words:
            try:
                footer_title = (clip_data.get("viral_title") or clip_data.get("title") or "") if clip_data else ""
                _generate_ass_subtitles(
                    words=filtered_words,
                    video_duration=clip_duration,
                    layout=layout,
                    caption_position=caption_position,
                    footer_title=footer_title,
                    output_ass_path=ass_path,
                    card_top_y=card_y,
                    card_bottom_y=card_bottom_y,
                )
                ass_generated = os.path.exists(ass_path)
            except Exception as e:
                logger.warning(f"Could not pre-generate ASS subtitles: {e}")

        # --- ULTRA-FAST SINGLE-PASS FFMPEG PIPELINE (5-6x FASTER) ---
        # Performs frame-accurate seek, 9:16 layout/card, ASS captions, and audio fade in ONE pass!
        try:
            logger.info("\n--- STEP 2: Executing Single-Pass Fast Render (Trim + No-Crop Card + Captions) ---")
            fade_start = max(0.0, clip_duration - 0.35)
            afade_filter = f"afade=t=out:st={fade_start:.2f}:d=0.35"
            ass_filter_str = f"ass={ass_rel_path}:fontsdir=assets/fonts" if ass_generated else "null"

            if layout == "card":
                mask_path = str(temp_dir_path / "card_mask.png")
                _create_rounded_mask_file(card_w, card_h, 28, mask_path)

                # Pure scale into card dimensions - PRESERVES 100% OF TEXT & VIDEO CONTENT (NO CROPPING!)
                card_filter = (
                    f"[0:v]scale={card_w}:{card_h}[v_scaled];"
                    f"[v_scaled][1:v]alphamerge[v_rounded];"
                    f"color=c=black:s={CANVAS_WIDTH}x{CANVAS_HEIGHT}:d={clip_duration:.2f}[bg];"
                    f"[bg][v_rounded]overlay={card_x}:{card_y}[card];"
                    f"[card]{ass_filter_str}[outv]"
                )
                if has_audio:
                    card_filter += f";[0:a]{afade_filter}[outa]"

                fast_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{clip_start:.3f}",
                    "-t", f"{clip_duration:.3f}",
                    "-i", video_path,
                    "-i", mask_path,
                    "-filter_complex", card_filter,
                    "-map", "[outv]",
                ]
                if has_audio:
                    fast_cmd.extend(["-map", "[outa]", "-c:a", AUDIO_CODEC, "-b:a", "192k"])
                fast_cmd.extend([
                    "-c:v", VIDEO_CODEC, "-preset", "veryfast", "-crf", "18",
                    "-movflags", "+faststart",
                    final_path
                ])
            else:
                target_width = height * 9 // 16
                crop_offset = (width - target_width) // 2
                full_filter = f"[0:v]crop={target_width}:{height}:{crop_offset}:0,scale={CANVAS_WIDTH}:{CANVAS_HEIGHT},{ass_filter_str}[outv]"
                if has_audio:
                    full_filter += f";[0:a]{afade_filter}[outa]"

                fast_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{clip_start:.3f}",
                    "-t", f"{clip_duration:.3f}",
                    "-i", video_path,
                    "-filter_complex", full_filter,
                    "-map", "[outv]",
                ]
                if has_audio:
                    fast_cmd.extend(["-map", "[outa]", "-c:a", AUDIO_CODEC, "-b:a", "192k"])
                fast_cmd.extend([
                    "-c:v", VIDEO_CODEC, "-preset", "veryfast", "-crf", "18",
                    "-movflags", "+faststart",
                    final_path
                ])

            _run_subprocess(fast_cmd, cancel_token)

            if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
                os.replace(final_path, output_path)
                filesize = os.path.getsize(output_path)
                logger.info(f"\n⚡ Ultra-Fast Single-Pass Short exported: {output_path}")
                logger.info(f"File size: {filesize / (1024*1024):.2f} MB")
                return output_path

        except Exception as single_pass_err:
            logger.warning(f"Single-pass FFmpeg engine had an issue, falling back to multi-step: {single_pass_err}")
        finally:
            if os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except OSError:
                    pass

        try:
            # Fallback Step 1: Trim video using ffmpeg
            logger.info("\n--- STEP 1 (Fallback): Trimming video ---")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            _run_ffmpeg(
                (
                    ffmpeg
                    .input(video_path, ss=clip_start, t=clip_duration)
                    .output(
                        trimmed_path,
                        vcodec=VIDEO_CODEC,
                        acodec=AUDIO_CODEC,
                        preset="veryfast",
                        movflags='+faststart',
                    )
                    .overwrite_output()
                ),
                cancel_token,
            )
            logger.info(f"Trimmed video saved to: {trimmed_path}")
            
            # Step 2: Get video dimensions for proper cropping
            logger.info("\n--- STEP 2: Analyzing video dimensions ---")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            probe = ffmpeg.probe(trimmed_path)

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise VideoProcessingError("Could not find video stream in trimmed file")
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            logger.info(f"Video dimensions: {width}x{height}")

            # Step 3: Frame into Card Style (Black BG + Rounded Box) or Full Crop
            if layout == "card":
                logger.info("\n--- STEP 3: Framing video into Card Style (Black BG + Rounded Box) ---")
                mask_path = str(temp_dir_path / "card_mask.png")
                _create_rounded_mask_file(card_w, card_h, 28, mask_path)

                card_filter = (
                    f"[0:v]scale={card_w}:{card_h}[v_scaled];"
                    f"[v_scaled][1:v]alphamerge[v_rounded];"
                    f"color=c=black:s={CANVAS_WIDTH}x{CANVAS_HEIGHT}:d={clip_duration}[bg];"
                    f"[bg][v_rounded]overlay={card_x}:{card_y}[outv]"
                )

                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", trimmed_path,
                    "-i", mask_path,
                    "-filter_complex", card_filter,
                    "-map", "[outv]",
                    "-map", "0:a",
                    "-c:v", VIDEO_CODEC, "-preset", "veryfast", "-crf", "18",
                    "-c:a", AUDIO_CODEC, "-b:a", "192k",
                    "-movflags", "+faststart",
                    cropped_path,
                ]
                _run_subprocess(cmd, cancel_token)
                logger.info(f"Card video framed to {CANVAS_WIDTH}x{CANVAS_HEIGHT}: {cropped_path}")
            else:
                # Full vertical crop fallback
                target_width = height * 9 // 16
                crop_offset = (width - target_width) // 2
                logger.info(f"Target Shorts ratio: {target_width}x{height}")
                logger.info("\n--- STEP 3: Cropping to 9:16 ---")

                crop_input = ffmpeg.input(trimmed_path)
                _run_ffmpeg(
                    (
                        ffmpeg
                        .output(
                            crop_input.video.filter(
                                'crop', target_width, height, crop_offset, 0
                            ),
                            crop_input.audio,
                            cropped_path,
                            vcodec=VIDEO_CODEC,
                            acodec=AUDIO_CODEC,
                            movflags='+faststart',
                        )
                        .overwrite_output()
                    ),
                    cancel_token,
                )
                logger.info(f"Cropped video saved to: {cropped_path}")

            # Verify audio stream exists in cropped video
            if not _has_audio_stream(cropped_path, cancel_token):
                logger.warning("⚠️ Cropped video has NO audio stream! Final Short may be silent.")

            # Filter and adjust transcript for clip
            filtered_words = _filter_words_for_clip(
                transcript_words,
                clip_start,
                clip_end,
                transcript_full_text,
            )

            # --- ULTRA-FAST HARDWARE ACCELERATED FFMPEG ENGINE (30-50x FASTER) ---
            # Renders karaoke subtitles directly inside FFmpeg with libass in ~5-8 seconds
            if filtered_words:
                ass_filename = f"captions_{int(time.time())}_{os.getpid()}.ass"
                ass_path = str(OUTPUT_DIR / ass_filename)
                ass_rel_path = f"output/{ass_filename}"
                try:
                    logger.info("\n--- STEP 4: Rendering Ultra-Fast Captions via FFmpeg libass Engine ---")
                    footer_title = (clip_data.get("viral_title") or clip_data.get("title") or "") if clip_data else ""

                    _generate_ass_subtitles(
                        words=filtered_words,
                        video_duration=clip_duration,
                        layout=layout,
                        caption_position=caption_position,
                        footer_title=footer_title,
                        output_ass_path=ass_path,
                    )

                    ass_filter = f"ass={ass_rel_path}:fontsdir=assets/fonts"
                    fade_start = max(0.0, clip_duration - 0.35)
                    afade_filter = f"afade=t=out:st={fade_start:.2f}:d=0.35"

                    fast_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", cropped_path,
                        "-vf", ass_filter,
                        "-af", afade_filter,
                        "-c:v", VIDEO_CODEC, "-preset", "veryfast", "-crf", "18",
                        "-c:a", AUDIO_CODEC, "-b:a", "192k",
                        "-movflags", "+faststart",
                        final_path
                    ]
                    _run_subprocess(fast_cmd, cancel_token)

                    if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
                        os.replace(final_path, output_path)
                        filesize = os.path.getsize(output_path)
                        logger.info(f"\n⚡ Fast Short exported in seconds: {output_path}")
                        logger.info(f"File size: {filesize / (1024*1024):.2f} MB")
                        return output_path
                except Exception as fast_err:
                    logger.warning(f"Fast FFmpeg subtitle engine had an issue, falling back to MoviePy: {fast_err}")
                finally:
                    if os.path.exists(ass_path):
                        try:
                            os.remove(ass_path)
                        except OSError:
                            pass

            # Fallback to MoviePy if fast engine is bypassed or had an issue
            logger.info("\n--- STEP 4 (Fallback): Loading video with MoviePy ---")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            try:
                video_clip = VideoFileClip(cropped_path)
                video_duration = video_clip.duration

            except PipelineCancelledError:
                raise
            except Exception as e:
                raise VideoProcessingError(f"Failed to load video with MoviePy: {e}") from e

            # Step 5: Processing captions with MoviePy
            logger.info("\n--- STEP 5 (Fallback): Processing MoviePy captions ---")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            if not filtered_words:
                logger.warning("No words found within clip timeframe")
                final_clip = video_clip
            else:
                sample = _caption_font_sample(filtered_words)
                font = _get_available_fonts(sample, cancel_token)
                text_clips = _create_caption_clips(
                    filtered_words,
                    video_duration,
                    font,
                    int(video_clip.h),
                    int(video_clip.w),
                    cancel_token,
                    video_clip=video_clip,
                    caption_position=caption_position,
                    layout=layout,
                )

                if not text_clips:
                    logger.warning("No valid text clips created, exporting video without captions")
                    final_clip = video_clip
                else:
                    # Composite video with captions
                    logger.info("\n--- STEP 6: Compositing video with captions ---")

                    if cancel_token is not None:
                        cancel_token.raise_if_cancelled()

                    footer_clips = []
                    if layout == "card" and clip_data:
                        title_raw = clip_data.get("viral_title") or clip_data.get("title") or ""
                        if title_raw:
                            clean_title = title_raw.split("|")[0].split("!")[0].strip()
                            if len(clean_title) > 36:
                                clean_title = clean_title[:36] + "..."
                            try:
                                foot_clip = _create_card_footer_clip(clean_title, font, CARD_FOOTER_FONT_SIZE, video_duration)
                                footer_clips.append(foot_clip)
                            except Exception as e:
                                logger.warning(f"Failed to create footer clip: {e}")

                    final_clip = CompositeVideoClip([video_clip] + text_clips + footer_clips)
            
            # Step 7: Export final video to a temporary path first.  This keeps
            # the previous successful output intact if the user cancels midway.
            logger.info("\n--- STEP 7: Exporting final Short ---")

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            export_logger = (
                _CancellableProgressLogger(cancel_token)
                if cancel_token is not None
                else None
            )

            final_clip.write_videofile(
                final_path,
                codec=VIDEO_CODEC,
                audio=False,
                remove_temp=True,
                threads=0,  # Let ffmpeg auto-detect optimal thread count
                preset="veryfast",  # Much faster encoding than "medium"
                logger=export_logger,
            )

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # Verify output before publishing it to the requested path.
            if not os.path.exists(final_path):
                raise VideoProcessingError("Export failed - file not created")

            # MoviePy was asked to write video only.  Remux the original audio
            # here so cancellation can also interrupt this final FFmpeg step.
            _ensure_audio_in_output(final_path, cropped_path, cancel_token)

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # Publish only a complete, valid output.
            os.replace(final_path, output_path)
            filesize = os.path.getsize(output_path)
            logger.info(f"\nFinal Short exported: {output_path}")
            logger.info(f"File size: {filesize / (1024*1024):.2f} MB")

            return output_path
            
        except PipelineCancelledError:
            logger.info("Video processing cancelled; partial output will be discarded")
            raise
        except ffmpeg.Error as e:
            raise VideoProcessingError(f"ffmpeg error: {e}") from e
        finally:
            # Cleanup MoviePy resources even when export is cancelled.
            if final_clip is not None and final_clip is not video_clip:
                try:
                    final_clip.close()
                except Exception as close_error:
                    logger.debug(f"Could not close composite clip: {close_error}")
            if video_clip is not None:
                try:
                    video_clip.close()
                except Exception as close_error:
                    logger.debug(f"Could not close video clip: {close_error}")
            for text_clip in text_clips:
                try:
                    text_clip.close()
                except Exception:
                    pass


def create_short_from_files(
    video_path: str,
    transcript_path: str,
    clip_data_path: str,
    output_path: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> str:
    """
    Convenience function to load all data from files and create the short.
    
    Args:
        video_path: Path to downloaded video
        transcript_path: Path to Phase 1 transcript JSON
        clip_data_path: Path to Phase 2 clip recommendation JSON
        output_path: Optional custom output path
        cancel_token: Optional token used to stop processing immediately
    
    Returns:
        Path to the final exported short video
    """
    logger.info("="*60)
    logger.info("PHASE 3: The Editor - Creating YouTube Short")
    logger.info("="*60)
    
    # Load transcript
    logger.info(f"\nLoading transcript from: {transcript_path}")
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Get word-level segments
    transcript_words = transcript_data.get("segments", [])
    
    # Older transcript files may not contain word-level timings.  Repair them
    # from the segment text (or the full transcript) so captions are never
    # silently skipped.
    ensure_word_timestamps(transcript_words, transcript_data.get("full_text", ""))
    
    # Load clip data
    logger.info(f"Loading clip data from: {clip_data_path}")
    with open(clip_data_path, 'r', encoding='utf-8') as f:
        clip_data = json.load(f)
    
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    return create_short(
        video_path,
        transcript_words,
        clip_data,
        output_path,
        cancel_token,
        transcript_data.get("full_text", ""),
    )


# CLI Entry Point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 3: Video Editor - Create YouTube Short with Animated Captions"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to downloaded video from Phase 1"
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript JSON from Phase 1"
    )
    parser.add_argument(
        "--clip-data",
        required=True,
        help="Path to clip recommendation JSON from Phase 2"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for final short (default: output/final_short.mp4)"
    )
    
    args = parser.parse_args()
    
    try:
        result_path = create_short_from_files(
            video_path=args.video,
            transcript_path=args.transcript,
            clip_data_path=args.clip_data,
            output_path=args.output
        )
        
        print("\n" + "="*60)
        print("SHORT CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutput: {result_path}")
        
    except (VideoProcessingError, CaptionGenerationError) as e:
        logger.error(f"Processing failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)