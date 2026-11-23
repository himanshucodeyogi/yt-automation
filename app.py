"""
Phase 5: The Command Center - Web Dashboard

A FastAPI backend with WebSocket streaming for real-time pipeline progress.
Connects to Phases 1-4 to run the complete YouTube Shorts automation.

Author: YouTube Shorts Automation Pipeline
"""

import os
import sys
import json
import uuid
import logging
import asyncio
import shutil
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from pipeline_control import CancellationToken, PipelineCancelledError


class PipelineRequest(BaseModel):
    """Request model for pipeline execution."""
    url: str
    groq_api_key: str = ""
    caption_position: str = "below"
    layout: str = "card"
    model: str = "openai/gpt-oss-120b"
    channel_id: Optional[str] = "ask"
    privacy_status: Optional[str] = "private"


class YouTubeConfigForm(BaseModel):
    client_id: str
    client_secret: str
    project_id: Optional[str] = "yt-automation"


class YouTubeUploadRequest(BaseModel):
    video_filename: str
    channel_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    privacy_status: Optional[str] = "private"


class QuickDownloadRequest(BaseModel):
    url: str
    format: str = "video"  # "video" or "audio"


class BrainstormRequest(BaseModel):
    topic_or_text: str
    groq_api_key: str = ""
    style: str = "viral"


class CustomRenderRequest(BaseModel):
    video_filename: str
    start_time: float
    end_time: float
    layout: str = "card"
    caption_position: str = "below"
    custom_title: Optional[str] = None


class TranscribeRequest(BaseModel):
    video_filename: str
    groq_api_key: str = ""


# Backwards-compatible name used by earlier versions of the dashboard.
PipelineStoppedError = PipelineCancelledError


from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Import pipeline phases
try:
    from phase1_scraper import run_phase1_pipeline, DownloadError, TranscriptionError
    from phase2_brain import analyze_transcript_for_clip, GroqAPIError, ClipExtractionError
    from phase3_editor import create_short, VideoProcessingError
    from phase4_publisher import (
        run_phase4_pipeline,
        upload_short_to_youtube,
        upload_short_to_channel,
        connect_new_channel,
        list_connected_channels,
        remove_channel,
        is_client_secrets_configured,
        get_client_secrets_info,
        save_client_secrets_from_credentials,
        YouTubeUploadError,
        AuthenticationError,
    )
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEMPLATES_DIR = Path("templates")
OUTPUT_DIR = Path("output")
DOWNLOADS_DIR = Path("downloads")
STATIC_DIR = Path("static")

# Create directories if they don't exist
TEMPLATES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Create thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=2)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for broadcasting status updates."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.job_status: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(
        self,
        message: str,
        job_id: Optional[str] = None,
        level: str = "info",
    ):
        """Broadcast a message to all connected WebSockets."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "job_id": job_id,
            "level": level,
        }
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    def set_job_status(self, job_id: str, status: Dict[str, Any]):
        """Store job status for later retrieval."""
        self.job_status[job_id] = status


manager = ConnectionManager()

# Cancellation tokens let the API interrupt the currently running worker,
# including active FFmpeg child processes, instead of waiting for a phase to end.
job_tokens: Dict[str, CancellationToken] = {}
stop_flags: Dict[str, bool] = {}


# Async pipeline runner
async def run_complete_pipeline(
    video_url: str,
    groq_api_key: str,
    job_id: str,
    cancel_token: CancellationToken,
    caption_position: str = "below",
    layout: str = "card",
    model: str = "openai/gpt-oss-120b",
    channel_id: str = "ask",
    privacy_status: str = "private",
):
    """
    Run the complete 4-phase pipeline asynchronously.
    
    Uses ThreadPoolExecutor to run blocking operations without freezing the server.
    
    Args:
        video_url: YouTube video URL
        groq_api_key: Groq API key for Phase 2
        job_id: Unique job identifier for tracking
    """
    
    async def emit(message: str, level: str = "info"):
        """Emit a message through WebSocket."""
        await manager.broadcast(message, job_id, level)
        logger.info(f"[Job {job_id}] {message}")
    
    async def raise_if_stopped():
        """Raise immediately when the user requested cancellation."""
        if cancel_token.is_cancelled() or stop_flags.get(job_id):
            raise PipelineCancelledError("Pipeline stopped by user request")
    
    # Warn if no Groq key - transcription will use local Whisper (slow on CPU)
    if not groq_api_key:
        await emit("⚠️ No Groq API key provided - using local Whisper (transcription will be slower)")
        await emit("💡 Add a Groq API key for 5-10x faster cloud transcription")
    
    phase1_result = None
    phase2_result = None
    video_path = None
    transcript_path = None
    clip_data_path = None
    
    # Set initial job status
    manager.set_job_status(job_id, {"status": "running", "phase": "phase1", "message": "Starting pipeline..."})
    
    try:
        await raise_if_stopped()
        
        # Phase 1: Download and Transcribe
        await emit("🚀 PHASE 1: Starting video download & transcription...")
        manager.set_job_status(job_id, {"status": "running", "phase": "phase1", "message": "Downloading and transcribing video..."})
        
        loop = asyncio.get_event_loop()

        def on_phase1_progress(msg: str):
            asyncio.run_coroutine_threadsafe(emit(msg), loop)

        phase1_result = await loop.run_in_executor(
            executor,
            lambda: run_phase1_pipeline(
                video_url,
                groq_api_key=groq_api_key,
                cancel_token=cancel_token,
                progress_callback=on_phase1_progress,
            )
        )
        
        video_path = phase1_result['download_info']['filepath']
        transcript_path = phase1_result['transcript_path']
        
        await emit(f"✅ PHASE 1 COMPLETE: {phase1_result['download_info']['title'][:50]}...")
        
        # Surface the word count - the animated captions are word-by-word, so
        # a transcript without word timings would produce a caption-less clip.
        transcript_segments = phase1_result['transcript_info'].get('segments', [])
        word_total = sum(len(s.get('words') or []) for s in transcript_segments)
        await emit(
            f"📄 Transcript: {len(phase1_result['transcript_info']['full_text'])} characters, "
            f"{word_total} timed words"
        )
        if word_total == 0:
            await emit(
                "⚠️ No word-level timings in the transcript - captions will be missing",
                level="warning",
            )
        else:
            await emit(f"✅ Captions ready: {word_total} words", level="success")
        manager.set_job_status(job_id, {"status": "running", "phase": "phase2", "message": "Analyzing transcript for viral clip..."})
        
        await raise_if_stopped()
        
        # Phase 2: Find Viral Clip
        await emit("\n🧠 PHASE 2: Analyzing transcript for viral clip...")
        
        phase2_result = await loop.run_in_executor(
            executor,
            lambda: analyze_transcript_for_clip(
                phase1_result['transcript_info'],
                groq_api_key,
                model=model,
                cancel_token=cancel_token,
            )
        )
        
        # Save clip data
        stem_name = Path(video_path).stem
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_filename = f"{stem_name}_short_{ts_str}.mp4"
        custom_out = str(OUTPUT_DIR / short_filename)

        clip_data_path = str(OUTPUT_DIR / f"{stem_name}_short_{ts_str}_clip_data.json")
        with open(clip_data_path, 'w', encoding='utf-8') as f:
            json.dump(phase2_result, f, indent=2)

        # Also save base reference for quick lookups
        try:
            with open(OUTPUT_DIR / f"{stem_name}_clip_data.json", 'w', encoding='utf-8') as f:
                json.dump(phase2_result, f, indent=2)
        except Exception:
            pass
        
        duration = phase2_result['clip_end_time'] - phase2_result['clip_start_time']
        await emit(f"✅ PHASE 2 COMPLETE: Clip {duration:.1f}s long")
        await emit(f"🎯 Title: {phase2_result['viral_title'][:60]}...")
        manager.set_job_status(job_id, {"status": "running", "phase": "phase3", "message": "Creating Short with animated captions..."})
        
        await raise_if_stopped()
        
        # Phase 3: Create Short with Captions
        await emit("\n✂️ PHASE 3: Creating Short with animated captions...")
        
        short_path = await loop.run_in_executor(
            executor,
            lambda: create_short(
                video_path=video_path,
                transcript_words=phase1_result['transcript_info']['segments'],
                clip_data=phase2_result,
                output_path=custom_out,
                cancel_token=cancel_token,
                transcript_full_text=phase1_result['transcript_info'].get('full_text', ''),
                caption_position=caption_position,
                layout=layout,
            )
        )
        
        try:
            shutil.copy2(short_path, str(OUTPUT_DIR / "final_short.mp4"))
            with open(OUTPUT_DIR / "final_short_clip_data.json", 'w', encoding='utf-8') as f:
                json.dump(phase2_result, f, indent=2)
        except Exception:
            pass
        
        await emit(f"✅ PHASE 3 COMPLETE: Short created at {short_path}")

        await raise_if_stopped()

        short_web_path = f"/output/{short_filename}"

        # Determine if auto upload to a connected channel was requested
        should_upload = False
        target_channel = None
        if channel_id and channel_id not in ("ask", "none", ""):
            channels = list_connected_channels()
            matching = [c for c in channels if c.get("channel_id") == channel_id]
            if matching:
                should_upload = True
                target_channel = matching[0]

        if not should_upload:
            local_result = {
                "video_id": Path(short_path).stem,
                "video_filename": short_filename,
                "title": phase2_result.get("viral_title", "Untitled Short"),
                "tags": phase2_result.get("tags", []),
                "youtube_url": "",
                "video_path": short_web_path,
                "local_filepath": short_path,
                "ready_to_upload": True,
                "seo_description": phase2_result.get("seo_description", ""),
                "clip_data": phase2_result,
            }
            await emit(f"📁 Short saved locally: {short_path}")
            await emit("💡 Short is ready! Choose a YouTube channel in the results card to upload or download MP4.")
            manager.set_job_status(job_id, {
                "status": "completed",
                **local_result,
            })
            await emit("\n" + "=" * 60)
            await emit("🎉 PIPELINE COMPLETE! Short generated & ready for upload.")
            await emit("=" * 60)
            return

        ch_title = target_channel.get("title", channel_id)
        manager.set_job_status(job_id, {"status": "running", "phase": "phase4", "message": f"Uploading to YouTube ({ch_title})..."})
        
        # Phase 4: Upload to YouTube
        await emit(f"\n📤 PHASE 4: Uploading to YouTube channel: {ch_title}...")
        
        upload_result = await loop.run_in_executor(
            executor,
            lambda: upload_short_to_channel(
                channel_id=target_channel["channel_id"],
                video_path=short_path,
                title=phase2_result.get("viral_title", ""),
                description=phase2_result.get("seo_description", ""),
                tags=phase2_result.get("tags", []),
                privacy_status=privacy_status or "private",
                cancel_token=cancel_token,
            )
        )
        
        await emit(f"✅ PHASE 4 COMPLETE: Video uploaded to {ch_title}!")
        await emit(f"🔗 YouTube URL: {upload_result['youtube_url']}")
        
        # Mark job complete
        manager.set_job_status(job_id, {
            "status": "completed",
            "result": upload_result,
            "youtube_url": upload_result['youtube_url'],
            "video_url": upload_result['youtube_url'],
            "video_id": upload_result['video_id'],
            "video_path": short_web_path,
            "video_filename": short_filename,
            "title": phase2_result.get("viral_title", "Untitled Short"),
            "tags": phase2_result.get("tags", []),
            "seo_description": phase2_result.get("seo_description", ""),
            "clip_data": phase2_result,
        })
        
        await emit("\n" + "="*60)
        await emit("🎉 PIPELINE COMPLETE! Video uploaded successfully!")
        await emit("="*60)
        
    except PipelineCancelledError:
        await emit("🛑 Pipeline stopped. Active processing was cancelled.")
        manager.set_job_status(job_id, {
            "status": "stopped",
            "message": "Stopped by user",
        })
    except DownloadError as e:
        await emit(f"❌ DOWNLOAD ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except TranscriptionError as e:
        await emit(f"❌ TRANSCRIPTION ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except GroqAPIError as e:
        await emit(f"❌ GROQ API ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except ClipExtractionError as e:
        await emit(f"❌ CLIP EXTRACTION ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except VideoProcessingError as e:
        await emit(f"❌ VIDEO PROCESSING ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except YouTubeUploadError as e:
        await emit(f"❌ YOUTUBE UPLOAD ERROR: {e}")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except AuthenticationError as e:
        await emit(f"❌ AUTHENTICATION ERROR: {e}")
        await emit("💡 Make sure you have client_secrets.json in the project directory")
        await emit("💡 Download from: https://console.cloud.google.com/apis/credentials")
        manager.set_job_status(job_id, {"status": "failed", "error": str(e)})
    except Exception as e:
        if cancel_token.is_cancelled():
            await emit("🛑 Pipeline stopped. Active processing was cancelled.")
            manager.set_job_status(job_id, {
                "status": "stopped",
                "message": "Stopped by user",
            })
        else:
            await emit(f"❌ UNEXPECTED ERROR: {e}")
            logger.exception("Pipeline error")
            manager.set_job_status(job_id, {"status": "failed", "error": str(e)})


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("🚀 Starting Auto-Shorts Command Center")
    
    # Make sure the local PO Token server is running. Modern YouTube rejects
    # most downloads without a valid Proof-of-Origin token.
    try:
        from phase1_scraper import ensure_bgutil_server, COOKIES_FILE
        if ensure_bgutil_server():
            logger.info("✅ PO Token server ready")
        else:
            logger.warning(
                "⚠️ PO Token server unavailable - YouTube downloads may fail "
                "with 'Requested format is not available'"
            )
        if COOKIES_FILE.exists():
            logger.info(f"✅ cookies.txt found ({COOKIES_FILE.resolve()})")
        else:
            logger.warning(
                "⚠️ cookies.txt not found - YouTube may show a bot-check error. "
                "Export your logged-in YouTube cookies into the project folder."
            )
    except Exception as exc:
        logger.warning(f"Startup check failed: {exc}")
    
    yield
    logger.info("🛑 Shutting down Auto-Shorts Command Center")


app = FastAPI(
    title="YouTube Shorts Automation Pipeline",
    description="Auto-generate YouTube Shorts from long-form videos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and media folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")


# HTML Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard UI."""
    template_path = Path("templates") / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/system/status")
async def get_system_status():
    """Retrieve detailed system diagnostics and hardware engine status."""
    from phase1_scraper import _bgutil_server_running, COOKIES_FILE, _find_node, check_cookies_status
    
    bgutil_up = _bgutil_server_running()
    cookie_info = check_cookies_status()
    node_path = _find_node()
    ffmpeg_path = shutil.which("ffmpeg")
    
    outputs = list(OUTPUT_DIR.glob("*.mp4"))
    downloads = list(DOWNLOADS_DIR.glob("*.mp4"))
    
    return JSONResponse(content={
        "bgutil_running": bgutil_up,
        "bgutil_url": "http://127.0.0.1:4416",
        "cookies_present": cookie_info["exists"],
        "cookies_authenticated": cookie_info["authenticated"],
        "cookies_has_sapisid": cookie_info.get("has_sapisid", False),
        "cookies_has_login_info": cookie_info.get("has_login_info", False),
        "cookies_message": cookie_info.get("message", ""),
        "cookies_size_bytes": cookie_info["size_bytes"],
        "node_detected": bool(node_path),
        "node_path": node_path or "Not found",
        "ffmpeg_detected": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path or "Not found",
        "total_outputs": len(outputs),
        "total_downloads": len(downloads),
        "client_secrets_present": Path("client_secrets.json").exists(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


class CookieUpdateRequest(BaseModel):
    content: str


@app.post("/api/cookies/update")
async def update_cookies(req: CookieUpdateRequest):
    """Save or paste new cookies.txt content."""
    from phase1_scraper import COOKIES_FILE, check_cookies_status
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Cookie content cannot be empty")
    
    try:
        COOKIES_FILE.write_text(content, encoding="utf-8")
        status = check_cookies_status()
        logger.info(f"✅ cookies.txt updated ({status['size_bytes']} bytes, authenticated={status['authenticated']})")
        return JSONResponse(content={
            "success": True,
            "status": status,
            "message": "cookies.txt updated successfully!"
        })
    except Exception as e:
        logger.error(f"Failed to update cookies.txt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/restart-bgutil")
async def restart_bgutil_server():
    """Restart or ensure the background PO-Token server is operational."""
    from phase1_scraper import ensure_bgutil_server, _bgutil_server_running
    is_ready = ensure_bgutil_server()
    return JSONResponse(content={
        "success": is_ready,
        "running": _bgutil_server_running(),
        "message": "PO Token Server is running on port 4416" if is_ready else "Failed to start PO Token server"
    })


@app.get("/api/library")
async def get_library_items():
    """List all generated shorts and downloaded source videos with rich metadata."""
    outputs = []
    # All mp4 files in output directory
    all_mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Filter out final_short.mp4 unless it's the only file, so it doesn't appear as a duplicate
    target_mp4s = [f for f in all_mp4s if f.name != "final_short.mp4"]
    if not target_mp4s:
        target_mp4s = all_mp4s

    for f in target_mp4s:
        stat = f.stat()
        base_id = f.stem.split("_short")[0]
        clip_data = {}
        candidate_jsons = [
            OUTPUT_DIR / f"{f.stem}_clip_data.json",
            OUTPUT_DIR / f"{base_id}_short_clip_data.json",
            OUTPUT_DIR / f"{base_id}_clip_data.json",
            OUTPUT_DIR / "final_short_clip_data.json"
        ]
        for candidate in candidate_jsons:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as jf:
                        clip_data = json.load(jf)
                    break
                except Exception:
                    pass
        
        outputs.append({
            "filename": f.name,
            "url": f"/output/{f.name}",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %H:%M"),
            "title": clip_data.get("viral_title") or f.stem,
            "seo_description": clip_data.get("seo_description", ""),
            "tags": clip_data.get("tags", []),
            "clip_start": clip_data.get("clip_start_time", 0),
            "clip_end": clip_data.get("clip_end_time", 0),
            "clip_text": clip_data.get("clip_text", "")
        })

    downloads = []
    for f in sorted(DOWNLOADS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        has_transcript = (DOWNLOADS_DIR / f"{f.stem}_transcript.json").exists()
        downloads.append({
            "filename": f.name,
            "url": f"/downloads/{f.name}",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %H:%M"),
            "has_transcript": has_transcript,
            "transcript_url": f"/downloads/{f.stem}_transcript.json" if has_transcript else None
        })

    return JSONResponse(content={"outputs": outputs, "downloads": downloads})


@app.delete("/api/media/{folder}/{filename}")
async def delete_media_file(folder: str, filename: str):
    """Safely delete an output short or downloaded raw video."""
    if folder not in ("output", "downloads"):
        raise HTTPException(status_code=400, detail="Invalid folder specified")
    
    base_dir = OUTPUT_DIR if folder == "output" else DOWNLOADS_DIR
    target = (base_dir / Path(filename).name).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        raise HTTPException(status_code=400, detail="Forbidden file path")
    
    if target.exists():
        target.unlink()
        # Clean companion json metadata
        for companion in base_dir.glob(f"{target.stem}*.*"):
            if companion.exists() and companion != target:
                try:
                    companion.unlink()
                except Exception:
                    pass
        return JSONResponse(content={"success": True, "message": f"Successfully deleted {filename}"})
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/api/tools/quick-download")
async def tool_quick_download(request: QuickDownloadRequest):
    """Direct video or audio extraction tool."""
    from phase1_scraper import download_video
    if not request.url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")
    
    quality = (
        "bestaudio/best"
        if request.format == "audio"
        else "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
    )
    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(
            executor,
            lambda: download_video(request.url, quality=quality)
        )
        fname = Path(info.get("filepath", "")).name
        return JSONResponse(content={
            "success": True,
            "title": info.get("title", ""),
            "filename": fname,
            "duration": info.get("duration", 0),
            "url": f"/downloads/{fname}"
        })
    except Exception as e:
        logger.exception("Quick download failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/ai-brainstorm")
async def tool_ai_brainstorm(request: BrainstormRequest):
    """Generate viral hooks, titles, descriptions, and hashtags using Groq AI."""
    import groq
    api_key = request.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="Groq API key required for AI Hook generation")
    
    prompt = f"""You are a master YouTube Shorts viral strategist.
Analyze this topic/content and generate high-engagement metadata for YouTube Shorts:
Topic / Content: "{request.topic_or_text}"
Tone / Style: {request.style}

Respond with a strictly valid JSON object matching this schema:
{{
  "viral_hooks": [
    "First 3-second hook line 1",
    "Hook line 2",
    "Hook line 3",
    "Hook line 4",
    "Hook line 5"
  ],
  "high_ctr_titles": [
    "Viral Title 1 with emoji (max 50 chars)",
    "Viral Title 2",
    "Viral Title 3",
    "Viral Title 4",
    "Viral Title 5"
  ],
  "seo_description": "2-3 highly engaging sentences with viral keywords and a CTA",
  "hashtags": ["#Shorts", "#Viral", "#Trending", "#Keyword1", "#Keyword2", "#Keyword3", "#Keyword4", "#Keyword5"]
}}
Return ONLY valid JSON."""

    loop = asyncio.get_event_loop()
    def _call():
        client = groq.Groq(api_key=api_key)
        models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        last_e = None
        for m in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "You are an elite YouTube growth strategist. Output only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=800
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                last_e = e
                continue
        raise HTTPException(status_code=500, detail=f"AI Brainstorm failed: {last_e}")

    try:
        data = await loop.run_in_executor(executor, _call)
        return JSONResponse(content={"success": True, "data": data})
    except Exception as e:
        logger.exception("AI brainstorm failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/custom-render")
async def tool_custom_render(request: CustomRenderRequest):
    """Render a customized short clip with precise timestamps and Hormozi captions."""
    video_path = DOWNLOADS_DIR / request.video_filename
    if not video_path.exists():
        video_path = OUTPUT_DIR / request.video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video not found")
    
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")
    
    transcript_file = DOWNLOADS_DIR / f"{video_path.stem}_transcript.json"
    transcript_words = []
    transcript_full_text = ""
    if transcript_file.exists():
        try:
            with open(transcript_file, "r", encoding="utf-8") as tf:
                t_data = json.load(tf)
                transcript_words = t_data.get("segments", [])
                transcript_full_text = t_data.get("full_text", "")
        except Exception:
            pass

    ts_custom = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"custom_{video_path.stem}_{int(request.start_time)}s_{int(request.end_time)}s_{ts_custom}.mp4"
    out_path = str(OUTPUT_DIR / out_name)
    
    clip_data = {
        "clip_start_time": float(request.start_time),
        "clip_end_time": float(request.end_time),
        "clip_text": request.custom_title or "Custom Short Clip",
        "viral_title": request.custom_title or f"Short: {video_path.stem}",
        "seo_description": f"Custom clip created from {request.start_time:.1f}s to {request.end_time:.1f}s",
        "tags": ["Shorts", "CustomClip", "Viral"]
    }
    
    loop = asyncio.get_event_loop()
    try:
        final_file = await loop.run_in_executor(
            executor,
            lambda: create_short(
                video_path=str(video_path),
                transcript_words=transcript_words,
                clip_data=clip_data,
                output_path=out_path,
                transcript_full_text=transcript_full_text,
                caption_position=request.caption_position,
                layout=request.layout
            )
        )
        with open(OUTPUT_DIR / f"{Path(final_file).stem}_clip_data.json", "w", encoding="utf-8") as f:
            json.dump(clip_data, f, indent=2)

        return JSONResponse(content={
            "success": True,
            "filename": Path(final_file).name,
            "url": f"/output/{Path(final_file).name}",
            "title": clip_data["viral_title"]
        })
    except Exception as e:
        logger.exception("Custom render error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/transcribe")
async def tool_transcribe_media(request: TranscribeRequest):
    """Transcribe an audio or video file with word-level timestamps."""
    video_path = DOWNLOADS_DIR / request.video_filename
    if not video_path.exists():
        video_path = OUTPUT_DIR / request.video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    
    from phase1_scraper import transcribe_video, transcribe_with_groq, save_transcript
    loop = asyncio.get_event_loop()
    try:
        def _do_transcribe():
            if request.groq_api_key:
                return transcribe_with_groq(str(video_path), request.groq_api_key)
            else:
                return transcribe_video(str(video_path))
        
        result = await loop.run_in_executor(executor, _do_transcribe)
        out_json = save_transcript(result)
        return JSONResponse(content={
            "success": True,
            "transcript_path": out_json,
            "full_text": result.get("full_text", ""),
            "segment_count": len(result.get("segments", [])),
            "language": result.get("language", "unknown")
        })
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# YouTube Multi-Account & Upload Endpoints
# ============================================

@app.get("/api/youtube/config")
async def get_yt_config():
    """Check if Google OAuth client secrets is configured and get connected channels count."""
    info = get_client_secrets_info()
    channels = list_connected_channels()
    info["channels_count"] = len(channels)
    return JSONResponse(content=info)


@app.post("/api/youtube/config")
async def set_yt_config(config: YouTubeConfigForm):
    """Save Google OAuth Client ID and Secret directly."""
    try:
        save_client_secrets_from_credentials(
            client_id=config.client_id,
            client_secret=config.client_secret,
            project_id=config.project_id or "yt-automation"
        )
        return JSONResponse(content={
            "success": True,
            "message": "Google OAuth credentials saved successfully!"
        })
    except Exception as e:
        logger.exception("Failed to save credentials")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/youtube/config/upload-file")
async def upload_secrets_file(file: UploadFile = File(...)):
    """Upload client_secrets.json directly from file."""
    try:
        content = await file.read()
        parsed = json.loads(content.decode("utf-8"))
        if "installed" not in parsed and "web" not in parsed:
            raise ValueError("Invalid client_secrets.json format. Must contain 'installed' or 'web' section.")
        with open("client_secrets.json", "wb") as f:
            f.write(content)
        return JSONResponse(content={
            "success": True,
            "message": "client_secrets.json uploaded and activated successfully!"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")


@app.get("/api/youtube/channels")
async def get_yt_channels():
    """List all connected YouTube channels."""
    channels = list_connected_channels()
    return JSONResponse(content={"channels": channels})


@app.post("/api/youtube/channels/connect")
async def connect_yt_channel():
    """
    Launch interactive Google OAuth 2.0 flow to connect a new YouTube channel.
    Opens the user's browser, handles token exchange, and stores channel details.
    """
    if not is_client_secrets_configured():
        raise HTTPException(
            status_code=400,
            detail="Google OAuth is not configured yet. Please configure client_secrets.json or provide Client ID & Secret first."
        )

    loop = asyncio.get_event_loop()
    try:
        await manager.broadcast("🔐 Google Sign-In opened in your browser. Please select your account and grant YouTube access...", level="info")
        channel_info = await loop.run_in_executor(
            executor,
            lambda: connect_new_channel()
        )
        await manager.broadcast(f"🎉 Connected YouTube channel: {channel_info.get('title')} ({channel_info.get('handle') or channel_info.get('channel_id')})", level="info")
        return JSONResponse(content={
            "success": True,
            "message": f"Channel connected: {channel_info.get('title')}",
            "channel": channel_info
        })
    except Exception as e:
        logger.exception("Failed to connect channel")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/youtube/channels/{channel_id}")
async def delete_yt_channel(channel_id: str):
    """Disconnect and remove a stored YouTube channel."""
    try:
        remove_channel(channel_id)
        return JSONResponse(content={"success": True, "message": "Channel removed successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/youtube/upload")
async def upload_video_to_yt(req: YouTubeUploadRequest):
    """Upload a specific video file to a selected connected YouTube channel."""
    video_path = OUTPUT_DIR / req.video_filename
    if not video_path.exists():
        video_path = DOWNLOADS_DIR / req.video_filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file '{req.video_filename}' not found.")

    channels = list_connected_channels()
    target_channel = next((c for c in channels if c.get("channel_id") == req.channel_id), None)
    if not target_channel:
        raise HTTPException(status_code=400, detail=f"Channel ID '{req.channel_id}' is not connected.")

    title = req.title or ""
    desc = req.description or ""
    tags = req.tags or []

    # If missing, attempt reading companion JSON
    clip_json = OUTPUT_DIR / f"{video_path.stem}_clip_data.json"
    if (not title or not desc or not tags) and clip_json.exists():
        try:
            with open(clip_json, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                if not title:
                    title = cdata.get("viral_title") or cdata.get("title", "")
                if not desc:
                    desc = cdata.get("seo_description") or cdata.get("description", "")
                if not tags:
                    tags = cdata.get("tags", [])
        except Exception:
            pass

    if not title:
        title = video_path.stem.replace("_", " ").title()

    loop = asyncio.get_event_loop()
    ch_title = target_channel.get("title", req.channel_id)

    try:
        await manager.broadcast(f"📤 Uploading {video_path.name} to {ch_title}...", level="info")

        def _progress(pct):
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(f"📤 Uploading: {pct}%...", level="info"),
                loop
            )

        result = await loop.run_in_executor(
            executor,
            lambda: upload_short_to_channel(
                channel_id=req.channel_id,
                video_path=str(video_path),
                title=title,
                description=desc,
                tags=tags,
                privacy_status=req.privacy_status or "private",
                progress_callback=_progress
            )
        )

        await manager.broadcast(f"🎉 Uploaded to {ch_title}! Video URL: {result.get('youtube_url')}", level="info")
        return JSONResponse(content={"success": True, **result})
    except Exception as e:
        logger.exception("Upload failed")
        await manager.broadcast(f"❌ YouTube upload error: {e}", level="warning")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a specific job."""
    status = manager.job_status.get(job_id, {"status": "not_found"})
    return JSONResponse(content=status)


@app.post("/api/start")
async def start_pipeline(request: PipelineRequest):
    """
    Start the complete pipeline.
    
    Request body as JSON:
    - url: YouTube video URL (required)
    - groq_api_key: Groq API key for cloud Whisper transcription (optional)
      If not provided, falls back to local Whisper (slower on CPU).
    """
    if not request.url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")
    
    # Basic URL validation
    if "youtube.com" not in request.url and "youtu.be" not in request.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Cancel any lingering previous jobs so they don't block executor threads
    for old_id, old_token in list(job_tokens.items()):
        old_status = manager.job_status.get(old_id, {}).get("status")
        if old_status in ("running", "queued", "starting"):
            try:
                old_token.cancel()
                stop_flags[old_id] = True
            except Exception:
                pass

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    cancel_token = CancellationToken()
    job_tokens[job_id] = cancel_token
    manager.set_job_status(job_id, {
        "status": "queued",
        "phase": "starting",
        "message": "Pipeline queued",
    })
    
    # Start pipeline in background
    asyncio.create_task(
        run_complete_pipeline(
            request.url,
            request.groq_api_key or "",  # Pass empty string if no key
            job_id,
            cancel_token,
            caption_position=request.caption_position or "below",
            layout=request.layout or "card",
            model=request.model or "openai/gpt-oss-120b",
            channel_id=request.channel_id or "ask",
            privacy_status=request.privacy_status or "private",
        )
    )
    
    return JSONResponse(content={
        "job_id": job_id,
        "status": "started",
        "message": f"Pipeline started for job {job_id}"
    })


@app.post("/api/stop/{job_id}")
async def stop_pipeline(job_id: str):
    """
    Stop a running pipeline job immediately.

    The cancellation token is observed inside every phase. Active FFmpeg
    processes are terminated as well, so video encoding does not continue in
    the background after this response.
    """
    if job_id not in manager.job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    current_status = manager.job_status.get(job_id, {})
    if current_status.get("status") in {"completed", "failed", "stopped"}:
        raise HTTPException(status_code=409, detail="Job is not running")

    cancel_token = job_tokens.get(job_id)
    if cancel_token is None:
        raise HTTPException(status_code=404, detail="Active job not found")

    # Publish the stopping state before cancelling. This prevents a fast
    # worker response from being overwritten by the "stopping" state.
    manager.set_job_status(job_id, {
        **current_status,
        "status": "stopping",
        "message": "Cancellation requested; stopping active work",
    })

    stop_flags[job_id] = True
    cancel_token.cancel()

    await manager.broadcast(
        "🛑 Stop requested. Cancelling active download, transcription, video encoding, or upload...",
        job_id,
    )

    return JSONResponse(content={
        "job_id": job_id,
        "status": "stopping",
        "message": f"Stop requested for job {job_id}"
    })


# WebSocket endpoint for real-time updates
@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for streaming pipeline progress."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive - wait for any message (ping/pong or client messages)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_connections": len(manager.active_connections)}


# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Auto-Shorts Command Center")
    print("="*60)
    print("\nStarting FastAPI server...")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60)
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False if os.environ.get("CODESPACES") else True,
        log_level="info"
    )