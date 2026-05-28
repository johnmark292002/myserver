import os
import uuid
import time
import threading
import io
from pathlib import Path
from queue import Queue
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, origins=['https://personal-filesarchive.netlify.app'])

UPLOAD_FOLDER = "uploads"
CLEANUP_INTERVAL = 300   # clean every 5 minutes
IMAGE_LIFETIME = 1800    # delete after 30 minutes
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}
MAX_IMAGE_SIZE = 1200     # resize long side to this value for speed

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Face detector (built‑in)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def resize_if_needed(img_np, max_dim=MAX_IMAGE_SIZE):
    """Resize image so that the longest side <= max_dim (preserves aspect ratio)"""
    h, w = img_np.shape[:2]
    if max(h, w) <= max_dim:
        return img_np
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def strong_enhancement(img_np):
    """
    Fast yet high‑quality enhancement:
    - Resize to reasonable dimensions
    - Bilateral filter (denoise + skin smoothing)
    - CLAHE contrast enhancement
    - Smart sharpening (only if needed)
    - Vibrance boost
    - Optional light face smoothing
    """
    # 1. Resize for speed
    img_small = resize_if_needed(img_np)

    # 2. Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)

    # 3. Bilateral filter (denoise + edge‑preserving smooth)
    smoothed = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=70, sigmaSpace=70)

    # 4. CLAHE (better local contrast)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Face detection + extra smoothing (optional, light)
    gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        face_roi = contrasted[y:y+h, x:x+w]
        face_smooth = cv2.bilateralFilter(face_roi, d=11, sigmaColor=60, sigmaSpace=60)
        contrasted[y:y+h, x:x+w] = cv2.addWeighted(face_smooth, 0.65, face_roi, 0.35, 0)

    # 6. Sharpening (medium strength to avoid artifacts)
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,   5, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(contrasted, -1, kernel)

    # 7. Vibrance (gentle saturation boost)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    final_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

# ----------------------- Job Queue -----------------------
jobs = {}
job_queue = Queue()
job_lock = threading.Lock()

def process_job(job):
    job_id = job['job_id']
    try:
        with job_lock:
            jobs[job_id]['status'] = 'processing'
        results = []
        for idx, (orig_name, img_bytes) in enumerate(job['images']):
            uid = f"{job_id}_{idx}"
            # Save original
            orig_ext = orig_name.rsplit('.', 1)[-1].lower()
            with open(os.path.join(UPLOAD_FOLDER, f"{uid}_original.{orig_ext}"), 'wb') as f:
                f.write(img_bytes)

            # Load and enhance
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(pil_img)
            enhanced_np = strong_enhancement(img_np)

            # Save enhanced as JPEG (95% quality)
            enhanced_pil = Image.fromarray(enhanced_np)
            enhanced_path = os.path.join(UPLOAD_FOLDER, f"{uid}_enhanced.jpg")
            enhanced_pil.save(enhanced_path, 'JPEG', quality=92, optimize=True)

            results.append({
                'original_url': f"/uploads/{uid}_original.{orig_ext}",
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

def worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        process_job(job)
        job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ----------------------- Cleanup -----------------------
def cleanup_old_files():
    now = time.time()
    for f in Path(UPLOAD_FOLDER).glob("*.*"):
        try:
            if now - f.stat().st_mtime > IMAGE_LIFETIME:
                f.unlink()
        except:
            pass

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
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file'}), 400
    file = request.files['image']
    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'}), 400

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
    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'error': 'No images uploaded'}), 400

    job_ids = []
    for file in files:
        if not file or file.filename == '' or not allowed_file(file.filename):
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
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    resp = {'job_id': job_id, 'status': job['status'], 'error': job.get('error')}
    if job['status'] == 'done':
        resp['results'] = job['results']
    return jsonify(resp)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if '..' in filename or filename.startswith('/'):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/')
def index():
    return send_from_directory('.', 'imageenhance.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
