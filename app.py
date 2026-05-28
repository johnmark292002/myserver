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
CLEANUP_INTERVAL = 60
IMAGE_LIFETIME = 3600
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Face detector (built-in)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def strong_enhancement(img_np):
    """
    Applies a professional pipeline without external models:
    - Denoising (non-local means)
    - CLAHE contrast enhancement
    - Bilateral filter (skin smoothing)
    - Face‑targeted smoothing
    - Strong sharpening
    - Vibrance boost
    """
    # 1. Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Denoise (non‑local means, preserves edges)
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

    # 3. CLAHE (improves local contrast)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Bilateral filter (smooths skin, keeps edges)
    smoothed = cv2.bilateralFilter(contrasted, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Face detection & extra smoothing
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        face_roi = smoothed[y:y+h, x:x+w]
        # Stronger bilateral filter on face
        face_smooth = cv2.bilateralFilter(face_roi, d=15, sigmaColor=80, sigmaSpace=80)
        # Blend: 70% smoothed, 30% original to avoid plastic look
        smoothed[y:y+h, x:x+w] = cv2.addWeighted(face_smooth, 0.7, face_roi, 0.3, 0)

    # 6. Strong sharpening kernel
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(smoothed, -1, kernel)

    # 7. Vibrance boost (HSV)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 12)  # increase saturation
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    final_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert back to RGB for PIL
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
            orig_ext = orig_name.rsplit('.',1)[-1].lower()
            with open(os.path.join(UPLOAD_FOLDER, f"{uid}_original.{orig_ext}"), 'wb') as f:
                f.write(img_bytes)
            # Load image with PIL and convert to numpy array
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_np = np.array(pil_img)
            # Enhance
            enhanced_np = strong_enhancement(img_np)
            # Convert back to PIL and save as JPEG quality 95
            enhanced_pil = Image.fromarray(enhanced_np)
            enhanced_path = os.path.join(UPLOAD_FOLDER, f"{uid}_enhanced.jpg")
            enhanced_pil.save(enhanced_path, 'JPEG', quality=95)
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

@app.route('/')
def index():
    return send_from_directory('.', 'imageenhance.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
