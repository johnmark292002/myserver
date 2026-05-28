import os
import uuid
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
import io

app = Flask(__name__)

# ✅ Your actual Netlify domain – already set
CORS(app, origins=['https://personal-filesarchive.netlify.app'])

UPLOAD_FOLDER = "uploads"
CLEANUP_INTERVAL = 60          # seconds
IMAGE_LIFETIME = 3600          # 1 hour
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_ai_enhancement(image: Image.Image) -> Image.Image:
    img = image.copy()
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.25)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)
    img = img.filter(ImageFilter.SHARPEN)
    return img

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

        enhanced = simulate_ai_enhancement(img)
        enhanced.save(os.path.join(UPLOAD_FOLDER, enh_name), 'JPEG', quality=92)

        return jsonify({
            'success': True,
            'original_url': f"/uploads/{orig_name}",
            'enhanced_url': f"/uploads/{enh_name}",
            'filename': enh_name,
            'original_name': file.filename
        })
    except Exception as e:
        print(e)
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
