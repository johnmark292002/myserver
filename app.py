import os
import uuid
import time
import threading
import io
import json
from pathlib import Path
from queue import Queue
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from PIL import Image

# Real-ESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import numpy as np

app = Flask(__name__)

# ✅ Allow your Netlify domain
CORS(app, origins=['https://personal-filesarchive.netlify.app'])

UPLOAD_FOLDER = "uploads"
CLEANUP_INTERVAL = 60
IMAGE_LIFETIME = 3600
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------- Job Queue -----------------------
jobs = {}                # job_id -> job info dict
job_queue = Queue()
job_lock = threading.Lock()

def process_job(job):
    """Worker thread function – runs Real-ESRGAN and face enhancement on a job."""
    job_id = job['job_id']
    results = []
    try:
        with job_lock:
            jobs[job_id]['status'] = 'processing'

        # ---------- Real-ESRGAN upscaler (x4) ----------
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=None,                     # automatically downloads pretrained model
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False                           # set True if GPU available
        )

        for idx, (orig_name, img_bytes) in enumerate(job['images']):
            # Save original
            uid = f"{job_id}_{idx}"
            with open(os.path.join(UPLOAD_FOLDER, f"{uid}_original.png"), 'wb') as f:
                f.write(img_bytes)

            # Load image with PIL and convert to numpy
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(pil_img)

            # Run Real-ESRGAN (super-resolution + enhancement)
            enhanced_np, _ = upsampler.enhance(img_np, outscale=4)  # 4x upscale

            # Face enhancement using built‑in face enhancement (requires facexlib/gfpgan)
            # We'll use RealESRGANer's face_enhance option by creating another instance, but
            # to keep it simple we'll just use the same instance (the model supports it if we
            # set the face_enhance flag). For production you'd use a dedicated face enhancer.
            # Instead, we'll apply a strong bilateral filter + sharpening on detected faces
            # as a fallback if no GPU face model is available. For now, let's use Real-ESRGAN's
            # face_enhance argument if we load the appropriate model.
            # Since we already have an upsampler without face_enhance, we'll do manual face boost:
            gray = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in faces:
                face_roi = enhanced_np[y:y+h, x:x+w]
                # Bilateral filter + sharpening on face
                face_roi = cv2.bilateralFilter(face_roi, d=9, sigmaColor=75, sigmaSpace=75)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                face_roi = cv2.filter2D(face_roi, -1, kernel)
                enhanced_np[y:y+h, x:x+w] = face_roi

            # Final output: convert back to PIL and save as high-quality JPEG
            enhanced_pil = Image.fromarray(enhanced_np)
            enhanced_filename = f"{uid}_enhanced.jpg"
            enhanced_pil.save(os.path.join(UPLOAD_FOLDER, enhanced_filename), 'JPEG', quality=95)

            results.append({
                'original_url': f"/uploads/{uid}_original.png",
                'enhanced_url': f"/uploads/{uid}_enhanced.jpg",
                'original_name': orig_name
            })

        with job_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['results'] = results

    except Exception as e:
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
        print(f"Job {job_id} failed: {e}")

def worker():
    """Continuously process jobs from the queue."""
    while True:
        job = job_queue.get()
        if job is None:
            break
        process_job(job)
        job_queue.task_done()

# Start worker thread
threading.Thread(target=worker, daemon=True).start()

# ----------------------- Cleanup Thread -----------------------
def cleanup_old_files():
    now = time.time()
    for f in Path(UPLOAD_FOLDER).glob("*.*"):
        try:
            if now - f.stat().st_mtime > IMAGE_LIFETIME:
                f.unlink()
                print(f"[Cleanup] Deleted: {f.name}")
        except: pass

def periodic_cleanup():
    while True:
        cleanup_old_files()
        time.sleep(CLEANUP_INTERVAL)

threading.Thread(target=periodic_cleanup, daemon=True).start()

# ----------------------- Routes -----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/enhance', methods=['POST'])
def enhance_single():
    """Upload a single image – returns a job ID."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    img_bytes = file.read()
    job_id = str(uuid.uuid4())
    with job_lock:
        jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'images': [(file.filename, img_bytes)],
            'results': None,
            'error': None
        }
    job_queue.put(jobs[job_id])
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/api/enhance/bulk', methods=['POST'])
def enhance_bulk():
    """Upload multiple images – returns a list of job IDs."""
    if 'images' not in request.files:
        # Fallback: 'images' key, or multiple files with same name
        files = request.files.getlist('images')
        if not files:
            return jsonify({'success': False, 'error': 'No images uploaded'}), 400
    else:
        files = [request.files['images']]  # single file scenario handled as list

    job_ids = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
        img_bytes = file.read()
        job_id = str(uuid.uuid4())
        with job_lock:
            jobs[job_id] = {
                'job_id': job_id,
                'status': 'pending',
                'images': [(file.filename, img_bytes)],
                'results': None,
                'error': None
            }
        job_queue.put(jobs[job_id])
        job_ids.append(job_id)

    return jsonify({'success': True, 'job_ids': job_ids})

@app.route('/api/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Poll job status and fetch results when done."""
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    resp = {
        'job_id': job_id,
        'status': job['status'],
        'error': job.get('error')
    }
    if job['status'] == 'done':
        resp['results'] = job['results']
    return jsonify(resp)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if '..' in filename or filename.startswith('/'):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def index():
    return send_from_directory('.', 'imageenhance.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
