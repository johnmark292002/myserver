import os
import uuid
import time
import threading
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
import cv2

app = Flask(__name__)

# ✅ Your Netlify domain
CORS(app, origins=['https://personal-filesarchive.netlify.app'])

UPLOAD_FOLDER = "uploads"
CLEANUP_INTERVAL = 60
IMAGE_LIFETIME = 3600
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Face detection setup -------------------
# Load OpenCV's Haar cascade for face detection (downloads once if missing)
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_PATH)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_ai_enhancement(image: Image.Image) -> Image.Image:
    """
    Professional AI enhancement pipeline:
    1. Convert to OpenCV (BGR)
    2. Non-local means denoising
    3. CLAHE (adaptive contrast)
    4. Bilateral filter (skin smoothing)
    5. Face-targeted smoothing (stronger bilateral on detected faces)
    6. Sharpening (unsharp mask + standard sharpen)
    7. Vibrance & colour enhancement
    8. Return as PIL Image
    """
    # Convert PIL -> NumPy (RGB) -> BGR for OpenCV
    img_np = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Denoising (non-local means, preserves edges)
    # Parameters: strength 10 (moderate), templateWindowSize 7, searchWindowSize 21
    denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)

    # 2. CLAHE on L channel (improve local contrast)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. Bilateral filter (smooth skin, keep edges)
    smoothed = cv2.bilateralFilter(contrasted, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Face detection & extra smoothing on faces
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = smoothed[y:y+h, x:x+w]
            # Stronger smoothing on face
            face_smooth = cv2.bilateralFilter(face_roi, d=15, sigmaColor=80, sigmaSpace=80)
            # Blend with original face to avoid plastic look (alpha 0.7 = 70% smoothed)
            blended = cv2.addWeighted(face_smooth, 0.7, face_roi, 0.3, 0)
            smoothed[y:y+h, x:x+w] = blended

    # 5. Sharpening
    # Create a sharpening kernel
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(smoothed, -1, kernel)

    # 6. Vibrance: convert to HSV, increase saturation slightly
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 10)  # increase saturation by 10 (scale 0-255)
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    final_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert back to PIL
    final_rgb = cv2.cvtColor(final_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

def cleanup_old_files():
    now = time.time()
    folder = Path(UPLOAD_FOLDER)
    for filepath in folder.glob("*.*"):
        if filepath.is_file():
            try:
                if now - filepath.stat().st_mtime > IMAGE_LIFETIME:
                    filepath.unlink()
                    print(f"[Cleanup] Deleted: {filepath.name}")
            except Exception as e:
                print(f"[Cleanup] Error deleting {filepath}: {e}")

def periodic_cleanup():
    while True:
        cleanup_old_files()
        time.sleep(CLEANUP_INTERVAL)

threading.Thread(target=periodic_cleanup, daemon=True).start()

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255,255,255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode=='RGBA' else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        uid = uuid.uuid4().hex
        orig_ext = file.filename.rsplit('.',1)[1].lower()
        orig_name = f"{uid}_original.{orig_ext}"
        enh_name = f"{uid}_enhanced.jpg"

        with open(os.path.join(UPLOAD_FOLDER, orig_name), 'wb') as f:
            f.write(img_bytes)

        # Run the powerful enhancement pipeline
        enhanced_img = simulate_ai_enhancement(img)
        enhanced_path = os.path.join(UPLOAD_FOLDER, enh_name)
        enhanced_img.save(enhanced_path, 'JPEG', quality=95)  # higher quality output

        return jsonify({
            'success': True,
            'original_url': f"/uploads/{orig_name}",
            'enhanced_url': f"/uploads/{enh_name}",
            'filename': enh_name,
            'original_name': file.filename
        })
    except Exception as e:
        print(f"Enhancement error: {e}")
        return jsonify({'success': False, 'error': 'Processing failed'}), 500

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
