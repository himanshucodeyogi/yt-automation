# YouTube Shorts Automation Pipeline - All 5 Phases

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/himanshucodeyogi/yt-automation)

A modular, 100% Free & Open-Source YouTube Shorts Automation Pipeline that:
1. Downloads trending YouTube videos
2. Transcribes with word-level timestamps using cloud/local AI
3. Finds the most viral 30-58s clip using Groq API
4. Creates animated captions with full story closure
5. Uploads to YouTube as Shorts with viral SEO
6. **Web Dashboard** for management, preview, and real-time monitoring

---

## ⚡ 1-Click Cloud Deployment (GitHub Codespaces - Free & 24/7)

1. Click the **[Open in GitHub Codespaces](https://codespaces.new/himanshucodeyogi/yt-automation)** button above.
2. Select default machine (**2-core, 8GB RAM - 100% Free**).
3. The container will automatically install FFmpeg, Devanagari fonts, and all Python/Node dependencies.
4. Once built, the web dashboard automatically launches on port `8000`.
5. Go to the **Ports** tab in Codespaces &rarr; Right click Port `8000` &rarr; Set **Port Visibility: Public**.
6. Copy the forwarded URL and open it from any device or send it to a friend!

---

## Architecture Status

- **PHASE 1 (Scraper & Listener)**: Download + Transcribe ✅ Completed
- **PHASE 2 (The Brain)**: AI clip extraction + SEO generation ✅ Completed
- **PHASE 3 (The Editor)**: Video cutting & animated captions ✅ Completed
- **PHASE 4 (The Publisher)**: YouTube auto-upload ✅ Completed
- **PHASE 5 (The Command Center)**: Web Dashboard ✅ **COMPLETE**

---

## Quick Start (Complete Pipeline)

### Web Dashboard (Recommended)

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
export GROQ_API_KEY="your-api-key"  # or: set GROQ_API_KEY=your-api-key

# 3. Start the web dashboard
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 4. Open http://localhost:8000 in your browser
# 5. Enter YouTube URL, save API key, click "Generate Short"
```

---

## ⚠️ YouTube Download Requirements (important)

Modern YouTube rejects most download attempts with
*"Sign in to confirm you're not a bot"*. Three things are needed:

| # | Requirement | Why |
|---|-------------|-----|
| 1 | **Node.js** (or Deno) | yt-dlp needs a JS runtime to solve YouTube's signature challenges |
| 2 | **Local PO Token server** | YouTube requires a "Proof of Origin" token for most player clients |
| 3 | **`cookies.txt`** in the project root | A logged-in session is what actually gets past the bot-check |

### 1. Install Node.js

Download from <https://nodejs.org/> — the app detects it automatically.

### 2. Build the PO Token server (one time)

```bash
cd tools/bgutil-server
npm install
npx tsc
```

The app **auto-starts** the server on `127.0.0.1:4416` when needed, so you
normally don't have to start it yourself. To run it manually:

```bash
cd tools/bgutil-server && node build/main.js --port 4416
```

### 3. Export `cookies.txt`

1. Log into YouTube in your browser
2. Install a "cookies.txt" browser extension (Netscape format)
3. Export the YouTube cookies and save the file as `cookies.txt` in the
   project root (next to `app.py`)

### Download strategy fallback chain

The downloader tries these in order and stops at the first success:

| Strategy | Client | Notes |
|----------|--------|-------|
| `cookies+mweb` | mweb | Official recommendation — best quality (up to 1080p/4K) |
| `cookies+default` | multi | Handles videos the mweb client rejects |
| `cookies+mweb-combined` | mweb | Single-file 360p fallback (always has audio) |
| `cookies+web_safari` | web_safari | HLS fallback |
| `cookies+android_vr` / `cookies+tv` | android_vr / tv | No PO token needed |
| `client:android_music` | android_music | Anonymous fallback |

Each strategy is retried twice, because a `HTTP 403` mid-download usually
just means the signed URL / PO token went stale — a fresh token fixes it.

---

### Command Line (All Phases)

```bash
# Install all dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your-api-key"

# Run complete pipeline via CLI
# Phase 1: Download and Transcribe
python phase1_scraper.py "https://youtube.com/watch?v=VIDEO_ID" --device cuda

# Phase 2: Find Viral Clip
python phase2_brain.py "downloads/VIDEOID_transcript.json"

# Phase 3: Create Short with Captions
python phase3_editor.py \
  --video downloads/VIDEOID.mp4 \
  --transcript downloads/VIDEOID_transcript.json \
  --clip-data output/clip_data.json

# Phase 4: Upload to YouTube
python phase4_publisher.py --privacy unlisted

# Or run all phases programmatically:
python main_pipeline.py
```

---

## Phase 1: Scraper & Listener

### `phase1_scraper.py`

**Downloads YouTube videos and generates word-level transcripts.**

| Function | Description |
|----------|-------------|
| `download_video(url)` | Downloads at 1080p using yt-dlp (cookies + PO token + fallback chain) |
| `transcribe_with_groq(video_path, api_key)` | Word-level transcript via Groq cloud Whisper (`whisper-large-v3`) |
| `transcribe_video(video_path)` | Word-level transcript via local faster-whisper (fallback) |
| `run_phase1_pipeline(video_url, groq_api_key=...)` | Complete orchestrator |

**Transcription:** pass a Groq API key and the audio is transcribed in the
cloud (5–10× faster than local Whisper on CPU). Without a key the pipeline
falls back to local `faster-whisper`. The full MP4 is never uploaded — only a
16 kHz mono MP3 (~0.7 MB for a 3-minute video) is sent to Groq.

**Word-level timings:** the animated captions are word-by-word, so every word
needs a timestamp. The pipeline requests `timestamp_granularities[]=word`
from Groq. Groq (like OpenAI) returns those timestamps in a **top-level
`words` array**, while `segments` only contains segment metadata — the parser
accepts that shape as well as the nested `segments[].words` shape some
OpenAI-compatible providers use. If a model returns no word timings at all,
they are estimated from the segment text (weighted by word length).
Transcripts are therefore never left without word timings — the dashboard
shows the timed-word count right after transcription.

**Output:**
- `downloads/{video_id}.{ext}` - Downloaded video
- `downloads/{video_id}_transcript.json` - Word-level transcript

```python
from phase1_scraper import run_phase1_pipeline

result = run_phase1_pipeline("https://youtube.com/watch?v=VIDEO_ID")
video_path = result['download_info']['filepath']
transcript_path = result['transcript_path']
```

---

## Phase 2: The Brain

### `phase2_brain.py`

**AI-powered viral clip detection and SEO generation using Groq's openai/gpt-oss-120b.**

| Function | Description |
|----------|-------------|
| `analyze_transcript_for_clip(transcript_data, groq_api_key)` | Finds 30-60s viral clip + SEO metadata |
| `run_phase2_pipeline(transcript_path, groq_api_key)` | File loader |

**Output:** `output/clip_data.json`

```python
from phase2_brain import run_phase2_pipeline

result = run_phase2_pipeline(transcript_path)
# result = {
#     "clip_start_time": 15.5,
#     "clip_end_time": 52.3,
#     "clip_text": "...",
#     "viral_title": "...",
#     "seo_description": "...",
#     "tags": [...]
# }
```

---

## Phase 3: The Editor

### `phase3_editor.py`

**Cuts video and adds Hormozi-style animated captions.**

| Function | Description |
|----------|-------------|
| `create_short(video_path, transcript_words, clip_data)` | Creates Short with captions |
| `create_short_from_files(...)` | Convenience loader |

**Key Features:**
- Exact timestamp trimming (ffmpeg)
- Center-crop to 9:16 (YouTube Shorts)
- **Multi-word captions**: a rolling window of 4 words stays on screen
- **Karaoke highlight**: the word that is being spoken right now is rendered
  ~35% bigger with a soft glow around it; when the next word starts the
  previous one shrinks back to the normal size and the new one lights up
- Hormozi styling: Yellow text, thick black stroke, bold font
- **Captions anchored near the bottom** (~13% margin from the bottom edge,
  scaled to the frame height — not centred)
- **No "tofu" boxes**: bundled Noto Sans / Noto Sans Devanagari fonts plus
  runtime glyph detection — a font that cannot render a character is skipped
  automatically, and undecodable `U+FFFD` characters are stripped
- **Captions rendered with PIL, not MoviePy `TextClip`**: MoviePy sizes its
  text canvas from font metrics that clip the bottom of the glyphs for the
  bundled Noto fonts (e.g. "WILL" rendered as "WT| |"). Captions are rendered
  to a tightly cropped RGBA bitmap and wrapped in an `ImageClip` instead, so
  every letter is complete and the position is pixel-exact.
- **Caption style is configurable** in `phase3_editor.py`:
  `CAPTION_GROUP_SIZE` (words on screen), `CAPTION_ACTIVE_SCALE` (how much
  bigger the spoken word is), `CAPTION_GLOW_RADIUS` / `CAPTION_GLOW_COLOR`
  (the halo) and `CAPTION_MAX_HOLD` (how long a line waits for the next
  word).
- **Audio is always preserved**: the crop step explicitly maps the audio
  stream, and a final remux re-attaches it if the export dropped it

**Output:** `output/final_short.mp4`

```python
from phase3_editor import create_short

short_path = create_short(
    video_path=video_path,
    transcript_words=transcript["segments"],
    clip_data=clip_data
)
```

---

## Phase 4: The Publisher

### `phase4_publisher.py`

**Uploads Shorts to YouTube using the Data API v3.**

| Function | Description |
|----------|-------------|
| `get_authenticated_service()` | OAuth 2.0 auth with token caching |
| `upload_short_to_youtube(youtube_client, video_path, clip_data)` | Upload to YouTube |
| `run_phase4_pipeline(video_path, clip_data_path)` | Complete orchestrator |

**OAuth Flow:**
- First run opens browser for authentication
- Token cached to `~/.yt-shorts-pipeline/token.pickle`
- Subsequent runs use cached credentials

**Local-only mode:** if `client_secrets.json` is missing, the dashboard skips
Phase 4 instead of failing the job — the finished Short stays at
`output/final_short.mp4` and the UI shows its local path. Add the credentials
file to enable automatic upload.

**Output:** Upload to YouTube with video ID

```python
from phase4_publisher import run_phase4_pipeline

result = run_phase4_pipeline(
    video_path="output/final_short.mp4",
    clip_data_path="output/clip_data.json",
    privacy_status="unlisted"  # private, unlisted, or public
)
# result = {
#     "video_id": "abc123",
#     "youtube_url": "https://youtu.be/abc123",
#     "title": "...",
#     "privacy_status": "unlisted"
# }
```

---

## Phase 5: The Command Center (Web Dashboard)

### `app.py` + `templates/index.html`

**Web-based dashboard with real-time progress tracking.**

| Feature | Description |
|---------|-------------|
| WebSocket streaming | Real-time log updates without page refresh |
| Job status tracking | Monitor pipeline progress |
| Immediate stop | Cancels the active phase and terminates running FFmpeg/encoding work |
| Local API key storage | Saves Groq key in browser localStorage |
| Dark mode | Sleek terminal-inspired dark theme |
| Results display | View video ID, URL, title, and tags |

**Run the dashboard:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

**API Endpoints:**
- `GET /` - Dashboard UI
- `POST /api/start` - Start pipeline (body: `{url, groq_api_key}`)
- `GET /api/status/{job_id}` - Get job status
- `POST /api/stop/{job_id}` - Stop the active job immediately (including FFmpeg encoding)
- `WebSocket /ws/status` - Real-time updates
- `GET /health` - Health check

---

## Google Cloud Setup (Phase 4)

**YouTube Data API v3 requires Google Cloud authentication:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project (or select existing)
3. Enable **YouTube Data API v3**
4. Create OAuth 2.0 Client ID:
   - Application type: **Desktop app**
   - Name: "YouTube Shorts Pipeline"
5. Download credentials as `client_secrets.json`
6. Place in project directory

**Quota Limits:**
- 10,000 units/day by default
- Each video upload: 1600 units
- ~6 uploads per day max (with default quota)

---

## File Structure

```
ytautomation/
├── app.py                    # FastAPI backend (Phase 5)
├── pipeline_control.py       # Shared cancellation and process termination
├── phase1_scraper.py         # Download + Transcribe
├── phase2_brain.py           # AI Clip + SEO
├── phase3_editor.py          # Video + Captions
├── phase4_publisher.py       # YouTube Upload
├── requirements.txt          # All dependencies
├── README.md                 # This file
├── cookies.txt               # Your logged-in YouTube cookies (Netscape format)
├── templates/
│   └── index.html            # Web UI (Phase 5)
├── assets/
│   └── fonts/                # Bundled Unicode fonts (no tofu boxes)
│       ├── NotoSans-Bold.ttf
│       └── NotoSansDevanagari-Bold.ttf
├── tools/
│   └── bgutil-server/        # Local PO Token server (Node.js)
├── downloads/                # Video files (auto-created)
│   ├── VIDEOID.mp4
│   └── VIDEOID_transcript.json
└── output/                   # Processing results (auto-created)
    ├── clip_data.json
    └── final_short.mp4
```

---

## Requirements

```bash
pip install -r requirements.txt

# Node.js is required for yt-dlp's JS challenge solver
# https://nodejs.org/

# Build the PO Token server (one time, from the project root)
cd tools/bgutil-server && npm install && npx tsc && cd ../..

# Optional: Install PyTorch for faster-whisper with GPU
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Phase 1 + 2 | Groq key for cloud Whisper transcription and AI clip extraction |
| `GROQ_API_BASE` | Optional | Custom Groq endpoint (if needed) |

The Groq key can also be typed into the dashboard (it is stored in the
browser's `localStorage` and sent with every request).

---

## Testing Each Phase

```bash
# Test Phase 1 (CPU mode)
python phase1_scraper.py "https://youtu.be/dQw4w9WgXcQ" --device cpu --model small

# Test Phase 2
export GROQ_API_KEY="your-key"
python phase2_brain.py "downloads/VIDEOID_transcript.json"

# Test Phase 3 (requires Phase 1 & 2 output)
python phase3_editor.py \
  --video downloads/VIDEOID.mp4 \
  --transcript downloads/VIDEOID_transcript.json \
  --clip-data output/clip_data.json

# Test Phase 4 (requires client_secrets.json from Google Cloud)
python phase4_publisher.py --privacy private

# Test Phase 5 (Web Dashboard)
uvicorn app:app --reload --port 8000
```

---

## Complete Python Pipeline Example

```python
import json

# Phase 1: Download and Transcribe
from phase1_scraper import run_phase1_pipeline
phase1 = run_phase1_pipeline("https://youtube.com/watch?v=VIDEO_ID")

# Phase 2: Find Viral Clip
from phase2_brain import run_phase2_pipeline
clip_data = run_phase2_pipeline(phase1['transcript_path'])

# Save clip data for Phase 3
with open("output/clip_data.json", "w") as f:
    json.dump(clip_data, f, indent=2)

# Phase 3: Create Short
from phase3_editor import create_short
with open(phase1['transcript_path']) as f:
    transcript = json.load(f)

short_path = create_short(
    video_path=phase1['download_info']['filepath'],
    transcript_words=transcript['segments'],
    clip_data=clip_data
)

# Phase 4: Upload to YouTube
from phase4_publisher import run_phase4_pipeline
upload_result = run_phase4_pipeline(short_path, "output/clip_data.json")

print(f"Short uploaded! Watch at: {upload_result['youtube_url']}")
```

---

## Limitations & Considerations

| Phase | Limitation | Mitigation |
|-------|------------|------------|
| Phase 1 | faster-whisper accuracy varies | Test different model sizes |
| Phase 2 | Groq API costs | Use openai/gpt-oss-120b (reasonable pricing) |
| Phase 3 | Font availability varies by OS | Auto-fallback to sans-serif |
| Phase 4 | YouTube API quota | Request quota increase if needed |

---

## License

This project is 100% Free & Open Source. Feel free to fork, modify, and use!

---

## Status: ✅ ALL PHASES COMPLETE

The complete YouTube Shorts Automation Pipeline is ready for use!

**Start the dashboard:**
```bash
uvicorn app:app --reload --port 8000
```

**Access:** http://localhost:8000