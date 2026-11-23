"""
Phase 1: YouTube Video Scraper & Transcriber

This module handles:
- Downloading YouTube videos using yt-dlp at 1080p quality
- Extracting word-level transcripts using faster-whisper (local execution)

Author: YouTube Shorts Automation Pipeline
"""

import os
import re
import json
import logging
import requests
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from pipeline_control import (
    CancellationToken,
    PipelineCancelledError,
    synthesize_word_timestamps,
    ensure_word_timestamps,
)

try:
    import yt_dlp
except ImportError:
    raise ImportError("yt_dlp is required. Install with: pip install yt-dlp")

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("faster-whisper is required. Install with: pip install faster-whisper")

# Groq Cloud Whisper config (fast transcription, no local GPU needed)
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"  # Fastest & most reliable Whisper on Groq (no 60-word truncation bug)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_DIR / "downloads"
# More robust format selection: tries 1080p, then 720p, then best available
# Uses format_sort to prioritize by quality rather than strict format matching
DEFAULT_VIDEO_QUALITY = "bv*[height<=1080]+ba/b[height<=1080]/bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"
DEFAULT_MODEL_SIZE = "small"  # faster-whisper model: tiny, small, medium, large-v3
DEFAULT_DEVICE = "cpu"  # Use "cpu" for broader compatibility; "cuda" if available and configured

# YouTube "Sign in to confirm you're not a bot" workarounds.
#
# Modern YouTube needs two things to download reliably:
#   1. A logged-in session (cookies.txt) - avoids the bot-check.
#   2. A "Proof of Origin" (PO) token for most player clients - without it
#      yt-dlp reports "Requested format is not available" even for videos it
#      can otherwise see. The official recommendation (yt-dlp PO Token Guide)
#      is the `mweb` client backed by a PO Token provider.
#
# The PO token provider plugin (bgutil-ytdlp-pot-provider) mints tokens via a
# small local HTTP server that we auto-start on demand.
BGUTIL_SERVER_DIR = Path("tools") / "bgutil-server"
BGUTIL_SERVER_URL = "http://127.0.0.1:4416"
BGUTIL_SERVER_PORT = 4416

# Player clients ordered by reliability.
#   mweb       - recommended by yt-dlp, works with a PO token provider
#   web_safari - serves HLS (m3u8) which needs no GVS PO token
#   android_vr / tv - do not require a PO token at all
YOUTUBE_CLIENTS = [
    "mweb",
    "web_safari",
    "android_vr",
    "android_music",
    "android_creator",
    "android",
    "tv",
    "ios",
    "web",
]

# Clients that need a PO token to expose their formats
PO_TOKEN_CLIENTS = ("mweb", "web", "web_safari", "web_music", "web_creator")

# Optional cookies file the user can drop in the project root
COOKIES_FILE = Path("cookies.txt")

# Error markers that indicate YouTube's bot-check
BOT_CHECK_MARKERS = (
    "Sign in to confirm",
    "confirm you're not a bot",
    "confirm you’re not a bot",
    "The page needs to be reloaded",
    "Request blocked",
    "HTTP Error 429",
)

# Error markers that indicate a missing / rejected PO token
PO_TOKEN_MARKERS = (
    "Requested format is not available",
    "requires a GVS PO Token",
    "HTTP Error 403",
)


@dataclass
class WordTimestamp:
    """Represents a single word with its timestamps."""
    word: str
    start: float
    end: float


@dataclass
class Segment:
    """Represents a transcript segment with words."""
    id: int
    start: float
    end: float
    words: List[WordTimestamp]
    text: str = ""


@dataclass
class TranscriptResult:
    """Complete transcript result with segments."""
    language: str
    segments: List[Segment]
    full_text: str


class DownloadError(Exception):
    """Raised when video download fails."""
    pass


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


def ensure_downloads_dir() -> Path:
    """
    Ensure the downloads directory exists.
    
    Returns:
        Path: The downloads directory path
    """
    downloads_path = DOWNLOADS_DIR.resolve()
    downloads_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloads directory: {downloads_path}")
    return downloads_path


def _find_node() -> Optional[str]:
    """
    Locate a Node.js executable.

    yt-dlp needs a JavaScript runtime to solve YouTube's JS challenges
    (signature deciphering). Deno is preferred by yt-dlp, but Node works too.
    """
    for name in ("deno", "node", "nodejs"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _bgutil_server_running() -> bool:
    """Check whether the local PO Token server is reachable."""
    try:
        response = requests.get(f"{BGUTIL_SERVER_URL}/ping", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def ensure_bgutil_server() -> bool:
    """
    Start the local PO Token provider server if it is not already running.

    The server lives in tools/bgutil-server and is started detached so it
    keeps running in the background between requests.

    Returns:
        True if the server is available (either already running or started)
    """
    if _bgutil_server_running():
        return True

    server_dir = BGUTIL_SERVER_DIR.resolve()
    main_js = server_dir / "build" / "main.js"
    node = _find_node()

    if not main_js.exists():
        logger.warning(
            "PO Token server not built (missing %s). "
            "Run: cd tools/bgutil-server && npm install && npx tsc",
            main_js,
        )
        return False

    if not node:
        logger.warning(
            "No JavaScript runtime (Node.js/Deno) found - PO tokens are unavailable."
        )
        return False

    logger.info("Starting local PO Token server on %s ...", BGUTIL_SERVER_URL)
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen(
            [
                node,
                str(main_js),
                "--port",
                str(BGUTIL_SERVER_PORT),
                "--host",
                "127.0.0.1",
            ],
            cwd=str(server_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except Exception as exc:
        logger.warning("Could not start PO Token server: %s", exc)
        return False

    # Wait for it to come up (max ~10s)
    for _ in range(20):
        time.sleep(0.5)
        if _bgutil_server_running():
            logger.info("PO Token server is ready")
            return True

    logger.warning("PO Token server did not become ready in time")
    return False


def _base_ydl_opts(
    quality: str,
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """Build the base yt-dlp options shared by every download strategy."""
    options = {
        # Output template: downloads/{video_id}.{ext}
        "outtmpl": str(DOWNLOADS_DIR / "%(id)s.%(ext)s"),
        
        # Format selection: best video/audio up to 1080p, merge into MP4
        "format": quality,
        
        # Format sorting: prefer higher resolution, then higher bitrate
        "format_sort": ["res:1080", "res:720", "res:480", "vcodec:avc", "acodec:mp4a"],
        
        # Prefer MP4 container
        "prefer_ffmpeg": True,
        "merge_output_format": "mp4",
        
        # Don't download subtitles (we'll transcribe from video)
        "writesubtitles": False,
        "writeautomaticsub": False,
        
        # Don't download metadata (we'll extract via API)
        "writeinfojson": False,
        
        # Quiet mode (we handle logging)
        "quiet": True,
        "no_warnings": True,
        
        # Playlist handling - just the single video
        "noplaylist": True,
        
        # Retry on transient failures
        "retries": 3,
        "fragment_retries": 3,
        
        # Network socket timeout prevents hanging on dead connections
        "socket_timeout": 20,
        
        # Fast streaming
        "sleep_interval": 0,
        "max_sleep_interval": 1,
    }

    # A JS runtime is required to solve YouTube's JS challenges
    node = _find_node()
    if node:
        options["js_runtimes"] = {"node": {"paths": [node]}}

    if cancel_token is not None:
        def _cancel_progress_hook(_status: Dict[str, Any]) -> None:
            cancel_token.raise_if_cancelled()

        options["progress_hooks"] = [_cancel_progress_hook]

    return options


def check_cookies_status() -> Dict[str, Any]:
    """Check whether cookies.txt exists and has logged-in authentication markers."""
    if not COOKIES_FILE.exists():
        return {
            "exists": False,
            "authenticated": False,
            "size_bytes": 0,
            "has_login_info": False,
            "has_sapisid": False,
            "message": "cookies.txt not found in root directory."
        }
    
    size = COOKIES_FILE.stat().st_size
    try:
        content = COOKIES_FILE.read_text(encoding="utf-8", errors="ignore")
        # Critical markers for a logged-in YouTube session
        has_sapisid = "SAPISID" in content or "__Secure-3PAPISID" in content
        has_login_info = "LOGIN_INFO" in content
        has_sid = "\tSID\t" in content or "\t__Secure-1PSID\t" in content or "\t__Secure-3PSID\t" in content
        
        is_auth = (has_sapisid or has_login_info) and has_sid
        return {
            "exists": True,
            "authenticated": is_auth,
            "has_login_info": has_login_info,
            "has_sapisid": has_sapisid,
            "size_bytes": size,
            "message": "Authenticated (Logged-in session)" if is_auth else "Guest/Unverified session (Risk of bot-check)"
        }
    except Exception as e:
        return {
            "exists": True,
            "authenticated": False,
            "size_bytes": size,
            "has_login_info": False,
            "has_sapisid": False,
            "message": f"Error reading cookies: {e}"
        }


def _build_download_strategies(
    quality: str,
    cancel_token: Optional[CancellationToken] = None,
) -> List[tuple]:
    """
    Build the ordered list of download strategies to try.
    
    Priority order:
      1. Logged-in session (cookies.txt) + `mweb` client with a PO Token.
         This is the officially recommended combination and the only one that
         reliably bypasses YouTube's bot-check *and* exposes HD formats.
      2. Same client but the *combined* single-file format. Separate
         adaptive video/audio streams sometimes get rejected with HTTP 403
         while the combined (360p) stream downloads fine.
      3. Logged-in session + clients that need no PO token.
      4. Anonymous fallbacks.
    
    Keeping this list short matters: hammering YouTube with many different
    clients in a row is what triggers IP-level bot-checks in the first place.
    
    Returns:
        List of (label, options) tuples
    """
    strategies = []
    
    cookie_path = COOKIES_FILE.resolve() if COOKIES_FILE.exists() else None
    
    # Isolate cookies in a temporary file so yt-dlp never pollutes or wipes cookies.txt on exit
    temp_cookie_path = None
    if cookie_path:
        try:
            temp_cookie_path = DOWNLOADS_DIR / f"_active_cookies.txt"
            shutil.copy2(cookie_path, temp_cookie_path)
        except Exception:
            temp_cookie_path = cookie_path
    
    script_path = PROJECT_DIR / "tools" / "bgutil-server" / "build" / "generate_once.js"

    def _make(
        client: Optional[str] = None,
        with_cookies: bool = False,
        format_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        opts = _base_ydl_opts(quality, cancel_token)
        if format_override:
            opts["format"] = format_override
        if with_cookies and temp_cookie_path:
            opts["cookiefile"] = str(temp_cookie_path)
        
        # Configure both HTTP and local script POT providers
        ext_args: Dict[str, Any] = {
            "youtubepot-bgutilhttp": {"base_url": [BGUTIL_SERVER_URL]},
        }
        if script_path.exists():
            ext_args["youtubepot-bgutilscript"] = {"script_path": [str(script_path.resolve())]}
        if client:
            ext_args["youtube"] = {"player_client": [client]}
        opts["extractor_args"] = ext_args
        return opts
    
    # 1. Authenticated + PO token capable (best quality, passes bot-check)
    if cookie_path:
        # Fastest + best quality when it works
        strategies.append(("cookies+mweb", _make("mweb", True)))
        # Multi-client fallback - handles videos the mweb client rejects (403)
        strategies.append(("cookies+default", _make(None, True)))
        # Single-file fallback: more reliable than separate adaptive streams
        # (lower resolution, but always contains audio)
        strategies.append(("cookies+mweb-combined", _make("mweb", True, "b")))
        strategies.append(("cookies+web_safari", _make("web_safari", True)))
        # Clients that do not require a PO token at all
        strategies.append(("cookies+android_vr", _make("android_vr", True)))
        strategies.append(("cookies+tv", _make("tv", True)))
    else:
        strategies.append(("client:mweb", _make("mweb", False)))
        strategies.append(("client:default", _make(None, False)))
        strategies.append(("client:mweb-combined", _make("mweb", False, "b")))
        strategies.append(("client:android_vr", _make("android_vr", False)))
    
    # 2. Anonymous fallbacks
    strategies.append(("client:android_music", _make("android_music", False)))
    strategies.append(("client:android_music-combined", _make("android_music", False, "b")))
    
    return strategies


def _cleanup_partial_files(video_id: str) -> None:
    """
    Remove leftover ``.part`` files for a video.
    
    yt-dlp resumes a partial download when the output file already exists.
    If we switch to a different player client the byte stream may differ, so
    a stale ``.part`` file could corrupt the final video. We therefore drop
    partials whenever we move on to a different strategy.
    """
    for pattern in (f"{video_id}*.part", f"{video_id}*.part-Frag*", f"{video_id}*.ytdl"):
        for part in DOWNLOADS_DIR.glob(pattern):
            try:
                part.unlink()
            except OSError:
                pass


def _is_bot_check(error: str) -> bool:
    """Check if an error message indicates YouTube's bot-check."""
    return any(marker.lower() in error.lower() for marker in BOT_CHECK_MARKERS)


def _is_po_token_issue(error: str) -> bool:
    """
    Check if an error indicates a missing / rejected Proof-of-Origin token.
    
    Modern YouTube requires a PO token for most player clients. Without it
    yt-dlp either sees no formats at all or gets HTTP 403 on the media URLs.
    """
    return any(marker.lower() in error.lower() for marker in PO_TOKEN_MARKERS)


def download_video(
    url: str,
    quality: str = DEFAULT_VIDEO_QUALITY,
    cancel_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Download a YouTube video using yt-dlp.
    
    Downloads the video at 1080p quality (or best available up to 1080p)
    and saves it to the downloads/ folder.
    
    YouTube frequently bot-checks yt-dlp requests ("Sign in to confirm
    you're not a bot"). To survive this we try several alternate player
    clients and browser cookies until one works.
    
    Args:
        url: YouTube video URL to download
        quality: Quality specification for yt-dlp (default: best up to 1080p)
        cancel_token: Optional token used to interrupt an active download
        progress_callback: Optional callback for streaming status messages
    
    Returns:
        Dict containing:
            - video_id: YouTube video ID
            - title: Video title
            - filepath: Path to downloaded video file
            - filesize: Size in bytes
            - format: Format string used
    
    Raises:
        DownloadError: If download fails for any reason
    """
    ensure_downloads_dir()

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    # Pre-check: If video already exists in downloads/, bypass download completely
    video_id_match = re.search(r'(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})', url)
    if video_id_match:
        vid = video_id_match.group(1)
        for ext in ("mp4", "mkv", "webm"):
            existing = DOWNLOADS_DIR / f"{vid}.{ext}"
            if existing.exists() and existing.stat().st_size > 1024 * 1024:
                size_mb = existing.stat().st_size / (1024 * 1024)
                logger.info(f"⚡ Video file already exists in downloads: {existing.name} ({size_mb:.1f} MB). Skipping YouTube download!")
                if progress_callback:
                    progress_callback(f"⚡ Video already cached locally: {existing.name} ({size_mb:.1f} MB)")
                return {
                    "video_id": vid,
                    "title": vid,
                    "filepath": str(existing),
                    "filesize": existing.stat().st_size,
                    "format": ext,
                }

    # Start the local PO Token server (needed by the `mweb`/`web` clients).
    # Without a PO token yt-dlp sees "Requested format is not available".
    if not ensure_bgutil_server():
        logger.warning(
            "PO Token server unavailable - falling back to clients that do not "
            "require a PO token (lower quality)"
        )

    strategies = _build_download_strategies(quality, cancel_token)
    last_error = ""
    bot_check_hit = False
    po_token_hit = False
    
    # A 403 usually means the PO token / signed URL went stale (it can happen
    # part-way through a large file). Re-running the same strategy fetches a
    # brand new token, so give every strategy a couple of attempts.
    MAX_ATTEMPTS_PER_STRATEGY = 2
    
    for index, (label, opts) in enumerate(strategies, start=1):
        current_video_id = "unknown"
        for attempt in range(1, MAX_ATTEMPTS_PER_STRATEGY + 1):
          try:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # Small delay between strategies to avoid hitting rate limits too fast
            if index > 1 and attempt == 1:
                time.sleep(1)
            elif attempt > 1:
                time.sleep(2)
            
            suffix = f" (attempt {attempt}/{MAX_ATTEMPTS_PER_STRATEGY})" if attempt > 1 else ""
            logger.info(
                f"Extracting video info for: {url} "
                f"[strategy {index}/{len(strategies)}: {label}]{suffix}"
            )
            if progress_callback:
                progress_callback(f"🔍 Fetching video info from YouTube [{label}]...")
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            
            if not info:
                last_error = "Could not extract info"
                logger.warning(f"  [{label}] no info returned, trying next strategy")
                break
            
            video_id = info.get("id", "unknown")
            current_video_id = video_id
            title = info.get("title", "Untitled")
            duration = info.get("duration", 0)
            
            # Log available formats for debugging
            formats = info.get("formats", [])
            video_formats = [f for f in formats if f.get("vcodec") != "none" and f.get("height")]
            if video_formats:
                available_heights = sorted(set(f.get("height") for f in video_formats if f.get("height")))
                logger.info(f"  Available video heights: {available_heights}")
            
            logger.info(f"  Video ID: {video_id}")
            logger.info(f"  Title: {title}")
            logger.info(f"  Duration: {duration} seconds")
            if progress_callback:
                progress_callback(f"🎬 Video: \"{title[:45]}...\" ({duration}s)")
                progress_callback(f"⬇️ Starting download with strategy [{label}]...")
            
            # Download the video with live progress hook
            download_opts = dict(opts)
            last_reported_pct = [-1]

            def _dl_hook(d):
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    if total > 0:
                        pct = int((downloaded / total) * 100)
                        if pct >= last_reported_pct[0] + 15 or pct == 100:
                            last_reported_pct[0] = pct
                            spd = d.get("_speed_str", "").strip()
                            eta = d.get("_eta_str", "").strip()
                            txt = f"⬇️ Downloading: {pct}%"
                            if spd:
                                txt += f" ({spd})"
                            if eta:
                                txt += f" [ETA: {eta}]"
                            if progress_callback:
                                progress_callback(txt)
                elif d.get("status") == "finished":
                    if progress_callback:
                        progress_callback("✅ Video download finished! Merging streams...")

            download_opts["progress_hooks"] = [_dl_hook]

            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.download([url])

            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            
            # Find the downloaded file
            video_files = list(DOWNLOADS_DIR.glob(f"{video_id}.*"))
            if not video_files:
                video_files = list(DOWNLOADS_DIR.glob(f"{video_id}*.*"))
            
            # Ignore leftovers from previous runs - only consider files that
            # were just written by this download.
            now = time.time()
            fresh_files = [
                vf for vf in video_files
                if vf.is_file() and (now - vf.stat().st_mtime) < 600
            ]
            if fresh_files:
                video_files = fresh_files
            
            if not video_files:
                last_error = "Downloaded file not found"
                logger.warning(f"  [{label}] file not found, trying next strategy")
                break
            
            # Prefer .mp4, otherwise the most recently written file
            video_path = None
            for vf in video_files:
                if vf.suffix.lower() == ".mp4":
                    video_path = vf
                    break
            
            if not video_path:
                video_path = max(video_files, key=lambda p: p.stat().st_mtime)
            
            filesize = video_path.stat().st_size
            
            logger.info(f"  Download complete: {video_path}")
            logger.info(f"  Filesize: {filesize / (1024*1024):.2f} MB")
            
            return {
                "video_id": video_id,
                "title": title,
                "filepath": str(video_path),
                "filesize": filesize,
                "format": quality,
                "duration": duration,
                "strategy": label,
            }
                
          except PipelineCancelledError:
            raise
          except yt_dlp.utils.DownloadError as e:
            if cancel_token is not None and cancel_token.is_cancelled():
                raise PipelineCancelledError("Video download cancelled by the user")
            last_error = str(e)
            if _is_bot_check(last_error):
                bot_check_hit = True
                logger.warning(f"  [{label}] bot-check blocked, trying next strategy")
                break
            if _is_po_token_issue(last_error):
                po_token_hit = True
                logger.warning(
                    f"  [{label}] PO token / format issue: {str(e)[:160]}"
                )
                # Retry the same strategy - a fresh PO token is fetched
                continue
            logger.warning(f"  [{label}] failed: {str(e)[:160]}, trying next strategy")
            break
          except Exception as e:
            if cancel_token is not None and cancel_token.is_cancelled():
                raise PipelineCancelledError("Video download cancelled by the user")
            last_error = str(e)
            logger.warning(f"  [{label}] unexpected error: {str(e)[:160]}, trying next strategy")
            break

        # All attempts for this strategy are exhausted - drop any partial
        # download so the next strategy starts from a clean slate.
        if current_video_id != "unknown":
            _cleanup_partial_files(current_video_id)
    
    # All strategies failed
    logger.error(f"All download strategies failed. Last error: {last_error}")
    
    if bot_check_hit:
        raise DownloadError(
            "YouTube blocked the download (bot-check). Tried all player clients and "
            "browser cookies. Fix by either:\n"
            "  1) Log in to YouTube in Chrome/Edge/Firefox on this PC (cookies auto-detected), or\n"
            "  2) Export cookies.txt (Netscape format) into the project folder."
        ) from DownloadError(last_error)
    
    if po_token_hit:
        raise DownloadError(
            "YouTube rejected the download because the PO Token was missing or "
            "expired (HTTP 403 / 'Requested format is not available').\n"
            "The local PO Token server should provide this automatically - make sure "
            "it is running:\n"
            "  cd tools/bgutil-server && node build/main.js --port 4416\n"
            "Last error: " + str(last_error)[:300]
        ) from DownloadError(last_error)
    
    raise DownloadError(f"Failed to download video: {last_error}")


def _word_from_data(word_data) -> WordTimestamp:
    """
    Convert faster-whisper word data to WordTimestamp dataclass.
    
    Args:
        word_data: Word object from faster-whisper (has .word, .start, .end attributes)
    
    Returns:
        WordTimestamp dataclass instance
    """
    return WordTimestamp(
        word=word_data.word,
        start=round(word_data.start, 3),
        end=round(word_data.end, 3)
    )


def _extract_audio_for_upload(
    video_path: str,
    cancel_token: Optional[CancellationToken] = None,
) -> Optional[str]:
    """
    Extract a small MP3 audio track for cloud transcription upload.

    The FFmpeg child process is registered with the cancellation token so the
    Stop button can terminate it immediately.
    """
    import imageio_ffmpeg

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if os.path.exists(ffmpeg_dir):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

    audio_path = os.path.join(
        tempfile.gettempdir(),
        f"groq_audio_{uuid.uuid4().hex[:8]}.mp3"
    )

    process = None

    try:
        process = subprocess.Popen(
            [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vn",                 # no video
                "-ac", "1",            # mono
                "-ar", "16000",        # 16kHz - what Whisper expects
                "-b:a", "64k",         # optimal speech bitrate (clear consonants)
                audio_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if cancel_token is not None:
            cancel_token.register_process(process)

        while True:
            try:
                process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
                output=process.stdout,
                stderr=process.stderr,
            )

        if os.path.exists(audio_path):
            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"Extracted audio for upload: {size_mb:.2f} MB")
            return audio_path
        return None

    except PipelineCancelledError:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            except OSError:
                pass
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        raise
    except Exception as e:
        logger.warning(f"Could not extract audio for upload: {e}")
        return None
    finally:
        if process is not None and cancel_token is not None:
            cancel_token.unregister_process(process)


def transcribe_with_groq(
    video_path: str,
    api_key: str,
    model: str = GROQ_WHISPER_MODEL,
    language: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio using Groq's cloud Whisper API.
    
    Much faster than local Whisper on CPU. Uses Groq's OpenAI-compatible
    Whisper endpoint with free tier limits.
    
    Args:
        video_path: Path to the video/audio file
        api_key: Groq API key
        model: Whisper model (whisper-large-v3 or whisper-large-v3-turbo)
        language: Language code (e.g., 'hi', 'en'). Auto-detected if None
        cancel_token: Optional token used to interrupt transcription
        
    Returns:
        Dict in the same format as local transcribe_video, or None on failure
    """
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    logger.info(f"Transcribing with Groq Whisper ({model}): {video_path}")
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "response_format": "verbose_json",
    }
    if language:
        data["language"] = language

    # Ask for word-level timestamps - the animated captions are word-by-word
    # so segment timestamps alone are not enough.  The API expects the
    # bracketed field name and repeated values:
    #     timestamp_granularities[]=word
    #     timestamp_granularities[]=segment
    data["timestamp_granularities[]"] = ["word", "segment"]
    
    # Extract a small audio-only file - uploading the full video (100MB+)
    # usually breaks the HTTPS connection
    audio_path = _extract_audio_for_upload(video_path, cancel_token)
    upload_path = audio_path or video_path
    upload_name = os.path.basename(upload_path)
    mime = "audio/mpeg" if audio_path else "video/mp4"
    
    def _post(payload: Dict[str, Any]):
        with open(upload_path, "rb") as f:
            files = {
                "file": (upload_name, f, mime)
            }
            return requests.post(
                GROQ_API_URL,
                headers=headers,
                files=files,
                data=payload,
                timeout=300
            )
    
    try:
        response = _post(data)
        
        # Some Groq models/deployments reject the granularities parameter.
        # Retry once without it - word timestamps are then synthesized from
        # the segment text so the captions still work.
        if response.status_code == 400 and "timestamp_granularities" in response.text:
            logger.warning(
                "Groq rejected timestamp_granularities - retrying without it "
                "(word timings will be estimated)"
            )
            data.pop("timestamp_granularities[]", None)
            response = _post(data)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Groq API request failed (will fallback to local Whisper): {e}")
        return None
    finally:
        # Clean up the temporary extracted audio
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
    
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    if response.status_code != 200:
        logger.warning(f"Groq API error {response.status_code} (will fallback to local Whisper): {response.text}")
        return None
    
    result = response.json()

    # Groq follows the OpenAI verbose_json schema: when word granularity is
    # requested, the timestamps arrive in a TOP-LEVEL "words" array while
    # "segments" only carries segment metadata.  Some compatible providers
    # nest the words under segments[].words instead, so accept both shapes.
    def _normalise_word(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        text = (raw.get("word") or raw.get("text") or "").strip()
        if not text:
            return None
        try:
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", start))
        except (TypeError, ValueError):
            return None
        if end < start:
            start, end = end, start
        # Clamp single word duration to realistic limit (prevents Whisper 17s glitch)
        max_duration = max(0.35, min(1.2, len(text) * 0.12))
        if (end - start) > max_duration:
            end = start + max_duration
        return {
            "word": text,
            "start": round(start, 3),
            "end": round(end, 3),
        }

    top_level_words = [
        w for w in (
            _normalise_word(raw)
            for raw in (
                result.get("words")
                if isinstance(result.get("words"), list)
                else []
            )
        ) if w
    ]

    raw_segments = (
        result.get("segments")
        if isinstance(result.get("segments"), list)
        else []
    )

    # Parse Groq response into our standard format
    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for index, seg in enumerate(raw_segments):
        seg_start = float(seg.get("start", 0) or 0)
        seg_end = float(seg.get("end", 0) or 0)
        seg_text = (seg.get("text") or "").strip()

        words = [
            w for w in (
                _normalise_word(raw)
                for raw in (seg.get("words") or [])
            ) if w
        ]

        segments.append({
            "id": seg.get("id") if seg.get("id") is not None else index,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg_text,
            "words": words,
        })
        if seg_text:
            full_text_parts.append(seg_text)

    # Attach the top-level words to the segments.  Per-segment words (when a
    # provider supplies them) win because they are already grouped correctly.
    if top_level_words and not any(seg["words"] for seg in segments):
        if not segments:
            # No segment metadata at all - group the words into short chunks
            # so Phase 2 still receives timestamped text.
            group: List[Dict[str, Any]] = []
            group_start = 0.0
            for word in sorted(top_level_words, key=lambda w: w["start"]):
                if group and (
                    word["start"] - group[-1]["end"] > 2.0
                    or word["start"] - group_start > 6.0
                    or len(group) >= 12
                ):
                    text = " ".join(w["word"] for w in group)
                    segments.append({
                        "id": len(segments),
                        "start": round(group_start, 3),
                        "end": round(group[-1]["end"], 3),
                        "text": text,
                        "words": group,
                    })
                    full_text_parts.append(text)
                    group = []
                if not group:
                    group_start = word["start"]
                group.append(word)
            if group:
                text = " ".join(w["word"] for w in group)
                segments.append({
                    "id": len(segments),
                    "start": round(group_start, 3),
                    "end": round(group[-1]["end"], 3),
                    "text": text,
                    "words": group,
                })
                full_text_parts.append(text)
        else:
            for word in top_level_words:
                midpoint = (word["start"] + word["end"]) / 2
                target = next(
                    (s for s in segments if s["start"] <= midpoint <= s["end"]),
                    None,
                )
                if target is None:
                    target = min(
                        segments,
                        key=lambda s: min(
                            abs(midpoint - s["start"]),
                            abs(midpoint - s["end"]),
                        ),
                    )
                target["words"].append(word)
            for seg in segments:
                seg["words"].sort(key=lambda w: w["start"])

    provider_words = sum(len(seg["words"]) for seg in segments)

    # Estimate timings only for segments that still have no words.  This is
    # the fallback for providers/models that do not support word granularity.
    synthesized_words = 0
    for seg in segments:
        if not seg["words"] and seg["text"]:
            seg["words"] = synthesize_word_timestamps(
                seg["text"], seg["start"], seg["end"]
            )
            synthesized_words += len(seg["words"])

    total_words = sum(len(seg["words"]) for seg in segments)
    full_text = " ".join(full_text_parts).strip()

    if provider_words:
        logger.info(f"  Word-level timestamps from Groq: {provider_words}")
    elif segments:
        logger.warning(
            "Groq returned no word-level timestamps (response keys: %s, "
            "top_level_words: %d) - estimating them from segment text",
            sorted(result.keys()),
            len(top_level_words),
        )

    if total_words == 0:
        logger.warning(
            "Groq returned no usable word timestamps and no segment text - "
            "captions will be empty"
        )
    elif synthesized_words:
        logger.info(
            f"  Estimated word timings for {synthesized_words} words "
            f"(API did not return word-level timestamps)"
        )

    logger.info(f"Groq transcription complete:")
    logger.info(f"  Language: {result.get('language', language or 'en')}")
    logger.info(f"  Duration: {result.get('duration', 0):.2f}s")
    logger.info(f"  Words: {total_words}")

    return {
        "language": result.get("language", language or "en"),
        "segments": segments,
        "full_text": full_text,
        "word_count": total_words,
        "video_path": video_path,
    }


def transcribe_video(
    video_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    language: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate word-level transcript using faster-whisper.
    
    Uses local Whisper models for transcription. Supports multiple model
    sizes (tiny, small, medium, large-v3) with word-level timestamps.
    
    Args:
        video_path: Path to the video file to transcribe
        model_size: Whisper model size (tiny, small, medium, large-v3)
        device: Device to use (cuda, cpu)
        language: Language code (e.g., 'en'). Auto-detected if None
        groq_api_key: Optional Groq API key for cloud transcription
        cancel_token: Optional token used to interrupt transcription
        progress_callback: Optional callback for streaming status messages
    
    Returns:
        Dict containing:
            - language: Detected language code
            - segments: List of segments, each with start/end times and words
            - full_text: Complete transcript as a single string
            - word_count: Total number of transcribed words
            - video_path: Path to the source video
    
    Raises:
        TranscriptionError: If transcription fails
    """
    # Validate input
    if not os.path.exists(video_path):
        raise TranscriptionError(f"Video file not found: {video_path}")

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    
    # Use Groq cloud Whisper if API key is provided (much faster on CPU systems)
    if groq_api_key:
        logger.info("Attempting Groq cloud Whisper transcription...")
        if progress_callback:
            progress_callback("🎙️ Transcribing with Groq Cloud Whisper API (lightning-fast)...")
        groq_result = transcribe_with_groq(
            video_path,
            groq_api_key,
            language=language,
            cancel_token=cancel_token,
        )
        if groq_result is not None:
            if progress_callback:
                progress_callback(f"✅ Groq Whisper transcription finished ({groq_result.get('word_count', 0)} words)")
            return groq_result
        logger.warning("Groq transcription failed, falling back to local Whisper...")
        if progress_callback:
            progress_callback("⚠️ Groq cloud failed, falling back to local Whisper...")
    else:
        if progress_callback:
            progress_callback(f"⚠️ No Groq API Key entered. Using local CPU Whisper '{model_size}' (takes 2-4 mins)...")

    logger.info(f"Loading Whisper model: {model_size} on {device}")
    if progress_callback:
        progress_callback(f"🧠 Loading local Whisper model '{model_size}'...")
    
    # Try CUDA first, then fallback to CPU
    devices_to_try = [device] if device == "cuda" else ["cpu"]
    if device == "cuda":
        devices_to_try.append("cpu")  # Fallback to CPU
    
    model = None
    for dev in devices_to_try:
        # Determine compute types for this device
        if dev == "cuda":
            compute_types_to_try = ["float16", "int8_float16", "int8"]
        else:
            compute_types_to_try = ["int8"]
        
        for compute_type in compute_types_to_try:
            try:
                logger.info(f"Attempting to load model on {dev} with compute_type: {compute_type}")
                model = WhisperModel(
                    model_size,
                    device=dev,
                    compute_type=compute_type
                )
                logger.info(f"Successfully loaded model on {dev} with compute_type: {compute_type}")
                break
            except ValueError as e:
                logger.warning(f"Failed to load on {dev} with compute_type {compute_type}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error on {dev} with compute_type {compute_type}: {e}")
                continue
        
        if model is not None:
            break
    
    if model is None:
        raise TranscriptionError(f"Failed to load Whisper model on any device/compute type combination")
    
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    logger.info(f"Transcribing: {video_path}")
    if progress_callback:
        progress_callback("🎙️ Processing speech and word-level timestamps...")
    
    # Try transcription, fallback to CPU if CUDA fails
    def try_transcribe(model_to_use, device_used):
        segments, info = model_to_use.transcribe(
            video_path,
            language=language,
            word_timestamps=True
        )
        
        # Convert segments to list
        segment_list = []
        full_text_parts = []
        total_words = 0
        last_report_t = [0]
        
        for segment in segments:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            # faster-whisper can return None for ``words`` even when
            # word_timestamps=True (e.g. on silence), so guard against it.
            words = [_word_from_data(w) for w in (segment.words or [])]
            total_words += len(words)
            
            # Keep the segment text as well - it lets the caption repair
            # logic rebuild word timings if the model returned none.
            segment_text = (getattr(segment, "text", "") or "").strip()
            if not segment_text:
                segment_text = " ".join(w.word for w in words)
            
            segment_obj = Segment(
                id=segment.id,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                words=words,
                text=segment_text,
            )
            segment_list.append(segment_obj)
            
            # Extract words for full text
            full_text_parts.append(segment_text)

            # Stream real-time progress to user so they know it is not frozen
            now_sec = time.time()
            if now_sec - last_report_t[0] >= 5 and getattr(info, "duration", 0) > 0:
                last_report_t[0] = now_sec
                pct = min(99, int((segment.end / info.duration) * 100))
                if progress_callback:
                    progress_callback(f"🎙️ Transcribing: {pct}% [{int(segment.end)}s / {int(info.duration)}s]...")
            
            # Keep the segment text as well - it lets the caption repair
            # logic rebuild word timings if the model returned none.
            segment_text = (getattr(segment, "text", "") or "").strip()
            if not segment_text:
                segment_text = " ".join(w.word for w in words)
            
            segment_obj = Segment(
                id=segment.id,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                words=words,
                text=segment_text,
            )
            segment_list.append(segment_obj)
            
            # Extract words for full text
            full_text_parts.append(segment_text)
        
        # Join all segments into full text
        full_text = " ".join(full_text_parts)
        
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": [asdict(s) for s in segment_list],
            "full_text": full_text.strip(),
            "word_count": total_words,
            "video_path": video_path,
        }
        
        logger.info(f"Transcription complete:")
        logger.info(f"  Language: {info.language}")
        logger.info(f"  Duration: {info.duration:.2f}s")
        logger.info(f"  Words: {total_words}")
        
        return result
    
    try:
        return try_transcribe(model, device)
    except PipelineCancelledError:
        raise
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower() or "cudnn" in str(e).lower():
            logger.warning(f"CUDA error during transcription: {e}")
            logger.info("Falling back to CPU...")
            # Load model on CPU
            cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            return try_transcribe(cpu_model, "cpu")
        else:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(f"Failed to transcribe video: {e}") from e
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise TranscriptionError(f"Failed to transcribe video: {e}") from e


def save_transcript(transcript: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Save transcript to JSON file.
    
    Args:
        transcript: Transcript result dictionary
        output_path: Optional custom output path. 
                     Defaults to downloads/{video_id}_transcript.json
    
    Returns:
        Path to saved JSON file
    """
    if output_path is None:
        # Extract video ID from path
        video_file = Path(transcript["video_path"])
        video_id = video_file.stem
        output_path = str(DOWNLOADS_DIR / f"{video_id}_transcript.json")
    
    ensure_downloads_dir()
    
    # Create a clean saveable version
    save_data = {
        "language": transcript["language"],
        "segments": transcript["segments"],
        "full_text": transcript["full_text"],
        "word_count": transcript["word_count"],
        "video_path": transcript["video_path"],
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transcript saved to: {output_path}")
    return output_path


def get_transcript_text(transcript: Dict[str, Any]) -> str:
    """
    Extract just the full text from transcript result.
    
    Args:
        transcript: Transcription result dictionary
    
    Returns:
        Full transcript text as a string
    """
    return transcript.get("full_text", "")


def run_phase1_pipeline(
    video_url: str,
    groq_api_key: Optional[str] = None,
    cancel_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 1 pipeline.
    
    Downloads a video and generates a word-level transcript.
    
    Args:
        video_url: YouTube video URL to process
        groq_api_key: Groq API key for cloud Whisper transcription (optional)
        cancel_token: Optional token used to interrupt the pipeline
        progress_callback: Optional callback for streaming status messages
    
    Returns:
        Dict containing:
            - download_info: Video download information
            - transcript_info: Transcription results
            - transcript_path: Path to saved transcript JSON
    
    Example:
        >>> result = run_phase1_pipeline("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> print(result['transcript_info']['full_text'])
    """
    logger.info("="*60)
    logger.info("PHASE 1: Scraper & Listener Pipeline")
    logger.info("="*60)
    
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    # Fast cache check: if this video and transcript were already fetched, reuse them instantly
    video_id_match = re.search(r'(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})', video_url)
    if video_id_match:
        vid = video_id_match.group(1)
        cached_video = DOWNLOADS_DIR / f"{vid}.mp4"
        cached_transcript = DOWNLOADS_DIR / f"{vid}_transcript.json"
        if cached_video.exists() and cached_video.stat().st_size > 1024 * 1024 and cached_transcript.exists():
            try:
                with open(cached_transcript, "r", encoding="utf-8") as f:
                    t_info = json.load(f)
                if t_info.get("segments") and t_info.get("full_text"):
                    logger.info(f"⚡ Video and transcript found in cache for [{vid}]. Skipping re-download & transcription!")
                    if progress_callback:
                        progress_callback(f"⚡ Cached video & transcript found for [{vid}]. Loading instantly!")
                    d_info = {
                        "video_id": vid,
                        "title": vid,
                        "filepath": str(cached_video),
                        "filesize": cached_video.stat().st_size,
                        "format": "cached",
                    }
                    return {
                        "download_info": d_info,
                        "transcript_info": t_info,
                        "transcript_path": str(cached_transcript),
                    }
            except Exception as e:
                logger.warning(f"Failed to load cached transcript for {vid}: {e}")

    # Step 1: Download video
    logger.info("\n--- STEP 1: Downloading Video ---")
    download_info = download_video(video_url, cancel_token=cancel_token, progress_callback=progress_callback)
    
    # Step 2: Transcribe video (Groq cloud if API key provided, else local Whisper)
    logger.info("\n--- STEP 2: Transcribing Video ---")
    transcript_info = transcribe_video(
        download_info["filepath"],
        groq_api_key=groq_api_key,
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    
    # Make sure every segment has word-level timings - the animated captions
    # are word-by-word and would otherwise be empty.
    ensure_word_timestamps(
        transcript_info.get("segments", []),
        transcript_info.get("full_text", ""),
    )
    transcript_info["word_count"] = sum(
        len(s.get("words") or []) for s in transcript_info.get("segments", [])
    )
    if transcript_info["word_count"] == 0:
        logger.warning(
            "Transcript contains no word-level timings - captions will be empty"
        )
    else:
        logger.info(f"Word-level timings ready: {transcript_info['word_count']} words")
    
    # Step 3: Save transcript
    logger.info("\n--- STEP 3: Saving Transcript ---")
    transcript_path = save_transcript(transcript_info)
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("="*60)
    
    return {
        "download_info": download_info,
        "transcript_info": transcript_info,
        "transcript_path": transcript_path,
    }


# CLI Entry Point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 1: YouTube Video Scraper & Transcriber"
    )
    parser.add_argument(
        "url",
        help="YouTube video URL to download and transcribe"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_SIZE,
        choices=["tiny", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help="Device to use for transcription (default: cuda)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_phase1_pipeline(args.url)
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Video: {result['download_info']['title']}")
        print(f"Downloaded to: {result['download_info']['filepath']}")
        print(f"Transcript saved to: {result['transcript_path']}")
        print(f"Language: {result['transcript_info']['language']}")
        print(f"Word count: {result['transcript_info']['word_count']}")
        print("\nFirst 200 characters of transcript:")
        print(result['transcript_info']['full_text'][:200] + "...")
        
    except DownloadError as e:
        logger.error(f"Download failed: {e}")
        exit(1)
    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)