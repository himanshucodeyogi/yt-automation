"""
Phase 4: The Publisher - YouTube Data API v3 Auto-Upload & Multi-Account Manager

This module handles:
- YouTube OAuth 2.0 authentication with multi-account management
- Connecting and switching between multiple YouTube channels
- Uploading videos to YouTube as Shorts with SEO optimization
- Safe token storage and automatic refresh

Author: YouTube Shorts Automation Pipeline
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from pipeline_control import CancellationToken, PipelineCancelledError

try:
    import google.auth.transport.requests
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError(
        "Google API dependencies are required. "
        "Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]
CLIENT_SECRETS_FILE = "client_secrets.json"
ACCOUNTS_DIR = Path("youtube_accounts")
CHANNELS_FILE = ACCOUNTS_DIR / "channels.json"
DEFAULT_CATEGORY_ID = "22"  # People & Blogs

# Upload chunk size (256KB - standard for YouTube)
CHUNK_SIZE = 256 * 1024

# Title length limit for YouTube
MAX_TITLE_LENGTH = 100


class YouTubeUploadError(Exception):
    """Raised when YouTube upload fails."""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


def ensure_accounts_dir() -> Path:
    """Ensure youtube_accounts directory exists."""
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    return ACCOUNTS_DIR


def is_client_secrets_configured(client_secrets_path: str = CLIENT_SECRETS_FILE) -> bool:
    """Check if client_secrets.json exists and is valid."""
    p = Path(client_secrets_path)
    if not p.exists():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return "installed" in data or "web" in data
    except Exception:
        return False


def get_client_secrets_info(client_secrets_path: str = CLIENT_SECRETS_FILE) -> Dict[str, Any]:
    """Return preview information about the configured client_secrets.json."""
    if not is_client_secrets_configured(client_secrets_path):
        return {"configured": False}
    try:
        with open(client_secrets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("installed") or data.get("web") or {}
        client_id = cfg.get("client_id", "")
        # Mask client_id preview for security
        masked_id = client_id[:12] + "..." + client_id[-8:] if len(client_id) > 20 else client_id
        return {
            "configured": True,
            "type": "installed" if "installed" in data else "web",
            "client_id_preview": masked_id,
            "project_id": cfg.get("project_id", "Default Project"),
        }
    except Exception as e:
        return {"configured": False, "error": str(e)}


def save_client_secrets_from_credentials(
    client_id: str,
    client_secret: str,
    project_id: str = "yt-automation",
    path: str = CLIENT_SECRETS_FILE
) -> bool:
    """
    Save client_secrets.json directly from Client ID and Client Secret strings.
    Formatted as Desktop application ('installed') credentials.
    """
    client_id = client_id.strip()
    client_secret = client_secret.strip()
    if not client_id or not client_secret:
        raise ValueError("Client ID and Client Secret cannot be empty")

    payload = {
        "installed": {
            "client_id": client_id,
            "project_id": project_id.strip() or "yt-automation",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
            "redirect_uris": [
                "http://localhost",
                "http://localhost:8080",
                "http://127.0.0.1",
            ]
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved client credentials to {path}")
    return True


def list_connected_channels() -> List[Dict[str, Any]]:
    """Return all connected YouTube channels from storage."""
    ensure_accounts_dir()
    if not CHANNELS_FILE.exists():
        return []
    try:
        with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
            channels = json.load(f)
            if isinstance(channels, list):
                return channels
            return []
    except Exception as e:
        logger.warning(f"Failed to read channels file: {e}")
        return []


def _save_channels_list(channels: List[Dict[str, Any]]):
    """Write channels list to storage."""
    ensure_accounts_dir()
    with open(CHANNELS_FILE, "w", encoding="utf-8") as f:
        json.dump(channels, f, indent=2)


def get_channel_info_from_client(youtube_client: Any) -> Dict[str, Any]:
    """Fetch channel profile info (title, customUrl, avatar, subscriberCount) using API."""
    try:
        response = youtube_client.channels().list(
            part="snippet,statistics",
            mine=True
        ).execute()

        items = response.get("items", [])
        if not items:
            raise ValueError("No YouTube channel found for this Google account.")

        channel = items[0]
        channel_id = channel.get("id")
        snippet = channel.get("snippet", {})
        statistics = channel.get("statistics", {})

        thumbnails = snippet.get("thumbnails", {})
        thumb_url = (
            thumbnails.get("default", {}).get("url") or
            thumbnails.get("medium", {}).get("url") or
            ""
        )

        return {
            "channel_id": channel_id,
            "title": snippet.get("title", "Unknown Channel"),
            "handle": snippet.get("customUrl", ""),
            "description": snippet.get("description", "")[:200],
            "thumbnail": thumb_url,
            "subscriber_count": statistics.get("subscriberCount", "0"),
            "video_count": statistics.get("videoCount", "0"),
            "view_count": statistics.get("viewCount", "0"),
            "connected_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to fetch channel profile: {e}")
        raise


def save_channel_credentials(channel_info: Dict[str, Any], creds: Any) -> None:
    """Save OAuth tokens and update channels registry."""
    ensure_accounts_dir()
    channel_id = channel_info["channel_id"]
    token_path = ACCOUNTS_DIR / f"{channel_id}.json"

    # Save credentials JSON
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    # Update channels list
    channels = list_connected_channels()
    # Remove existing entry if re-authenticating same channel
    channels = [c for c in channels if c.get("channel_id") != channel_id]
    channels.insert(0, channel_info)
    _save_channels_list(channels)
    logger.info(f"Successfully saved credentials and channel info for: {channel_info.get('title')} ({channel_id})")


def remove_channel(channel_id: str) -> bool:
    """Remove a connected YouTube channel and its stored credentials."""
    channels = list_connected_channels()
    new_channels = [c for c in channels if c.get("channel_id") != channel_id]
    _save_channels_list(new_channels)

    token_path = ACCOUNTS_DIR / f"{channel_id}.json"
    if token_path.exists():
        try:
            token_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete token file {token_path}: {e}")

    logger.info(f"Removed channel {channel_id}")
    return True


def get_authenticated_service_for_channel(
    channel_id: str,
    client_secrets_path: str = CLIENT_SECRETS_FILE,
    cancel_token: Optional[CancellationToken] = None,
) -> Any:
    """
    Authenticate and return YouTube API service for a specific stored channel ID.
    Automatically refreshes credentials if expired.
    """
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    token_path = ACCOUNTS_DIR / f"{channel_id}.json"
    if not token_path.exists():
        raise AuthenticationError(f"No credentials found for channel ID: {channel_id}")

    try:
        with open(token_path, "r", encoding="utf-8") as f:
            creds_data = json.load(f)
        creds = Credentials.from_authorized_user_info(creds_data, scopes=SCOPES)
    except Exception as e:
        raise AuthenticationError(f"Failed to load credentials for channel {channel_id}: {e}") from e

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        logger.info(f"Refreshing expired credentials for channel {channel_id}...")
        try:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            creds.refresh(google.auth.transport.requests.Request())
            # Save refreshed credentials
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            logger.info("Credentials refreshed and saved successfully.")
        except PipelineCancelledError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Could not refresh token for channel {channel_id}: {e}") from e

    if not creds.valid:
        raise AuthenticationError(f"Credentials for channel {channel_id} are invalid. Please re-authenticate.")

    service = build("youtube", "v3", credentials=creds)
    return service


def connect_new_channel(
    client_secrets_path: str = CLIENT_SECRETS_FILE,
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """
    Initiate interactive OAuth 2.0 flow to connect a new YouTube channel.
    Prompts user to select account and grants permission.
    Returns channel profile info.
    """
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    if not os.path.exists(client_secrets_path):
        raise FileNotFoundError(
            f"client_secrets.json not found at: {client_secrets_path}.\n"
            "Please configure your Google Cloud OAuth Client ID first."
        )

    logger.info("=" * 60)
    logger.info("Connecting New YouTube Channel (OAuth Flow)")
    logger.info("=" * 60)

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_path,
            scopes=SCOPES
        )
        # Use prompt='select_account' to allow user to choose or switch accounts/channels
        creds = flow.run_local_server(
            port=0,
            prompt="select_account",
            open_browser=True,
            authorization_prompt_message="Opening browser to sign in to YouTube: {url}"
        )
    except Exception as e:
        if cancel_token is not None and cancel_token.is_cancelled():
            raise PipelineCancelledError("Channel connection cancelled")
        raise AuthenticationError(f"OAuth flow failed: {e}") from e

    # Build service to fetch channel profile
    try:
        service = build("youtube", "v3", credentials=creds)
        channel_info = get_channel_info_from_client(service)
        save_channel_credentials(channel_info, creds)
        return channel_info
    except Exception as e:
        raise AuthenticationError(f"Failed to fetch profile or save channel: {e}") from e


def _prepare_title(title: str) -> str:
    """Prepare title for YouTube upload. Truncates if necessary and adds #Shorts if not present."""
    title = title.strip()
    if "#Shorts" not in title.lower():
        title = f"{title} #Shorts"
    if len(title) > MAX_TITLE_LENGTH:
        if title.lower().endswith("#shorts"):
            truncate_at = MAX_TITLE_LENGTH - len("#Shorts")
            title = title[:truncate_at].strip() + " #Shorts"
        else:
            title = title[:MAX_TITLE_LENGTH].strip()
    return title


def _prepare_description(description: str, tags: list) -> str:
    """Prepare description with hashtag integration from tags."""
    description = description.strip()
    hashtags = []
    for tag in tags[:10]:
        tag = tag.strip().replace(" ", "")
        if tag and not tag.startswith("#"):
            hashtags.append(f"#{tag}")
    if hashtags:
        hashtag_line = "\n\n" + " ".join(hashtags)
        description = description + hashtag_line
    return description


def upload_short_to_youtube(
    youtube_client: Any,
    video_path: str,
    clip_data: Dict[str, Any],
    privacy_status: str = "private",
    category_id: str = DEFAULT_CATEGORY_ID,
    cancel_token: Optional[CancellationToken] = None,
    progress_callback = None,
) -> Dict[str, Any]:
    """
    Upload a YouTube Short to YouTube using the Data API v3.
    """
    logger.info("=" * 60)
    logger.info("Uploading Short to YouTube")
    logger.info("=" * 60)

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()

    if not os.path.exists(video_path):
        raise YouTubeUploadError(f"Video file not found: {video_path}")

    raw_title = clip_data.get("viral_title") or clip_data.get("title") or "YouTube Short"
    raw_description = clip_data.get("seo_description") or clip_data.get("description") or ""
    raw_tags = clip_data.get("tags", [])

    title = _prepare_title(raw_title)
    if isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    else:
        tags = list(raw_tags)

    description = _prepare_description(raw_description, tags)

    if "shorts" not in [t.lower() for t in tags]:
        tags.append("shorts")

    privacy_status = privacy_status.lower()
    if privacy_status not in ["private", "unlisted", "public"]:
        privacy_status = "private"

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags[:30],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "embeddable": False,
            "license": "youtube",
            "selfDeclaredMadeForKids": False,
        }
    }

    video_size = os.path.getsize(video_path)
    logger.info(f"Video file: {video_path} ({video_size / (1024*1024):.2f} MB)")
    logger.info(f"Title: {title}")
    logger.info(f"Privacy: {privacy_status}")

    try:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        request = youtube_client.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=MediaFileUpload(
                video_path,
                chunksize=CHUNK_SIZE,
                resumable=True,
                mimetype="video/mp4"
            )
        )

        response = None
        progress = 0

        while response is None:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                if progress_callback:
                    try:
                        progress_callback(progress)
                    except Exception:
                        pass
                if progress % 20 == 0 or progress == 100:
                    logger.info(f"Upload progress: {progress}%")

        video_id = response.get("id")
        if not video_id:
            raise YouTubeUploadError("Upload succeeded but no video ID returned")

        logger.info("UPLOAD SUCCESSFUL!")
        logger.info(f"Video ID: {video_id}")
        logger.info(f"YouTube URL: https://youtu.be/{video_id}")

        return {
            "video_id": video_id,
            "title": title,
            "description": description,
            "tags": tags,
            "privacy_status": privacy_status,
            "embeddable": body["status"]["embeddable"],
            "youtube_url": f"https://youtu.be/{video_id}"
        }

    except PipelineCancelledError:
        raise
    except HttpError as e:
        error_details = ""
        try:
            error_json = json.loads(e.content.decode())
            error_details = error_json.get("error", {}).get("message", str(e))
        except Exception:
            error_details = str(e)
        raise YouTubeUploadError(f"YouTube API error: {error_details}") from e
    except Exception as e:
        if cancel_token is not None and cancel_token.is_cancelled():
            raise PipelineCancelledError("YouTube upload cancelled by the user") from e
        raise YouTubeUploadError(f"Upload failed: {e}") from e


def upload_short_to_channel(
    channel_id: str,
    video_path: str,
    title: str = "",
    description: str = "",
    tags: Optional[List[str]] = None,
    privacy_status: str = "private",
    cancel_token: Optional[CancellationToken] = None,
    progress_callback = None,
) -> Dict[str, Any]:
    """
    High-level entry point to upload a Short to a chosen channel ID.
    """
    logger.info(f"Uploading video to channel ID: {channel_id}")
    youtube_client = get_authenticated_service_for_channel(channel_id, cancel_token=cancel_token)

    clip_data = {
        "viral_title": title,
        "seo_description": description,
        "tags": tags or ["Shorts"],
    }

    result = upload_short_to_youtube(
        youtube_client=youtube_client,
        video_path=video_path,
        clip_data=clip_data,
        privacy_status=privacy_status,
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )
    result["channel_id"] = channel_id
    return result


# Backwards compatibility helper for legacy single-token flow
def get_authenticated_service(
    client_secrets_path: str = CLIENT_SECRETS_FILE,
    cancel_token: Optional[CancellationToken] = None,
) -> Any:
    """Legacy helper: gets authenticated service using first connected channel or default token."""
    channels = list_connected_channels()
    if channels:
        return get_authenticated_service_for_channel(channels[0]["channel_id"], client_secrets_path, cancel_token)

    # Fallback to single token flow if available
    token_path = Path.home() / ".yt-shorts-pipeline" / "token.pickle"
    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            return build("youtube", "v3", credentials=creds)

    # Otherwise connect new channel
    ch = connect_new_channel(client_secrets_path, cancel_token)
    return get_authenticated_service_for_channel(ch["channel_id"], client_secrets_path, cancel_token)


def run_phase4_pipeline(
    video_path: str,
    clip_data_path: str,
    client_secrets_path: str = CLIENT_SECRETS_FILE,
    privacy_status: str = "private",
    cancel_token: Optional[CancellationToken] = None,
) -> Dict[str, Any]:
    """Legacy wrapper for running Phase 4 upload."""
    if not os.path.exists(clip_data_path):
        raise FileNotFoundError(f"Clip data not found: {clip_data_path}")

    with open(clip_data_path, "r", encoding="utf-8") as f:
        clip_data = json.load(f)

    youtube_client = get_authenticated_service(client_secrets_path, cancel_token)
    return upload_short_to_youtube(
        youtube_client=youtube_client,
        video_path=video_path,
        clip_data=clip_data,
        privacy_status=privacy_status,
        cancel_token=cancel_token,
    )
