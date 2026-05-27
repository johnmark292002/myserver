import os
import uuid
import threading
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

# Load .env file only for local development – on Render we use environment variables
load_dotenv()

# Configure Cloudinary from environment variables
cloudinary.config(
    cloud_name=os.environ.get("CLOUD_NAME"),
    api_key=os.environ.get("API_KEY"),
    api_secret=os.environ.get("API_SECRET"),
    secure=True
)

app = Flask(__name__)
CORS(app)  # Allow any frontend origin (adjust in production if needed)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

# In-memory store for scheduled deletions (only for demo – on restart timers are lost)
pending_deletions = {}

# Resolution mapping (target width & height)
RESOLUTION_MAP = {
    "720": (1280, 720),
    "1080": (1920, 1080),
    "1440": (2560, 1440),
    "2160": (3840, 2160)   # 4K
}

# ----------------------------------------------------------------------
# Helper: build Cloudinary transformation string
# ----------------------------------------------------------------------
def build_transformation(active_opts: list, target_res: str) -> str:
    """
    active_opts: list of enabled stages, e.g. ['denoise','color','sharpen','upscale']
    target_res: '720', '1080', '1440' or '2160'
    Returns a Cloudinary transformation string like:
    "e_denoise:80/e_viesus_correct/e_sharpen:80/e_upscale/w_1920,h_1080,c_fill/f_auto/q_auto"
    """
    parts = []
    # Strict pipeline order: Denoise → Auto Color → Sharpen → Upscale
    if "denoise" in active_opts:
        parts.append("e_denoise:80")
    if "color" in active_opts:
        parts.append("e_viesus_correct")
    if "sharpen" in active_opts:
        parts.append("e_sharpen:80")
    if "upscale" in active_opts:
        parts.append("e_upscale")

    # Final dimensions (applied after upscale, or as a simple resize)
    width, height = RESOLUTION_MAP.get(target_res, (1920, 1080))
    parts.append(f"w_{width},h_{height},c_fill")   # fill to exact size
    parts.append("f_auto")   # auto format (webp/mp4)
    parts.append("q_auto")   # auto quality

    return "/".join(parts)

# ----------------------------------------------------------------------
# Schedule deletion after 1 hour
# ----------------------------------------------------------------------
def schedule_deletion(public_id: str, delay_seconds: int = 3600):
    """Delete the Cloudinary video after `delay_seconds` and clean up memory."""
    def delete_resource():
        try:
            cloudinary.uploader.destroy(public_id, resource_type="video")
            print(f"[Cleanup] Deleted {public_id}")
        except Exception as e:
            print(f"[Cleanup] Failed to delete {public_id}: {e}")
        finally:
            if public_id in pending_deletions:
                del pending_deletions[public_id]

    timer = threading.Timer(delay_seconds, delete_resource)
    timer.daemon = True
    timer.start()
    pending_deletions[public_id] = {"timer": timer, "deleted": False}
    return timer

# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Receive video file, upload to Cloudinary, build enhanced URL."""
    # Validate file
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not file.content_type.startswith("video/"):
        return jsonify({"error": "Only video files are allowed"}), 400

    # Get enhancement parameters
    resolution = request.form.get("resolution", "1080")
    options_str = request.form.get("options", "denoise,color,sharpen,upscale")
    active_opts = [opt.strip() for opt in options_str.split(",") if opt.strip()]

    # Validate resolution
    if resolution not in RESOLUTION_MAP:
        return jsonify({"error": f"Invalid resolution: {resolution}"}), 400

    try:
        # Upload video to Cloudinary (public_id = random UUID)
        public_id = str(uuid.uuid4())
        upload_result = cloudinary.uploader.upload(
            file,
            public_id=public_id,
            resource_type="video",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S UTC")
        )

        # Get original video metadata (width, height, etc.)
        resource_info = cloudinary.api.resource(public_id, resource_type="video")
        width = resource_info.get("width")
        height = resource_info.get("height")

        # Build enhanced URL with the requested pipeline
        transformation = build_transformation(active_opts, resolution)
        enhanced_url = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type="video",
            transformation=transformation,
            secure=True
        )[0]

        # Original (non‑transformed) URL
        original_url = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type="video",
            secure=True
        )[0]

        # Auto‑delete after 1 hour
        schedule_deletion(public_id, delay_seconds=3600)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        return jsonify({
            "public_id": public_id,
            "id": public_id,               # backend task ID
            "enhanced_url": enhanced_url,
            "original_url": original_url,
            "width": width,
            "height": height,
            "expires_at": expires_at,
            "message": "Enhancement pipeline ready"
        }), 200

    except cloudinary.exceptions.Error as e:
        return jsonify({"error": f"Cloudinary error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/api/task/<task_id>", methods=["DELETE"])
def delete_task(task_id):
    """Delete the video from Cloudinary and cancel scheduled deletion."""
    try:
        # Cancel the pending timer if it exists
        if task_id in pending_deletions:
            timer_obj = pending_deletions[task_id]["timer"]
            timer_obj.cancel()
            del pending_deletions[task_id]

        # Delete from Cloudinary
        result = cloudinary.uploader.destroy(task_id, resource_type="video")
        if result.get("result") == "ok":
            return jsonify({"status": "deleted", "id": task_id}), 200
        else:
            return jsonify({"error": "Resource not found or already deleted"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------
# Run the server (for local development; Render uses gunicorn)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)