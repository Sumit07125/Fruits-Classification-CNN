import io
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'fruit_resnet50.onnx'
CLASS_NAMES_PATH = BASE_DIR / 'models' / 'class_names.json'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
FALLBACK_CLASSES = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20000000

def create_app() -> Flask:
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES
    return app

app = create_app()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_fruit_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH.name}')
    
    # Load class names
    class_names = FALLBACK_CLASSES
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
            
    # Initialize ONNX Runtime session
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    return session, class_names

try:
    SESSION, CLASS_NAMES = load_fruit_model()
    MODEL_ERROR = None
    app.logger.info('Loaded %s with classes: %s', MODEL_PATH.name, CLASS_NAMES)
except Exception as exc:
    SESSION = None
    CLASS_NAMES = FALLBACK_CLASSES
    MODEL_ERROR = str(exc)
    app.logger.exception('Could not load the ONNX model')

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_uploaded_image(file_storage) -> Image.Image:
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError('The uploaded file is empty.')
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        return image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('Please upload a valid image file.') from exc

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize shortest edge to 256
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(256 * h / w)
    else:
        new_h = 256
        new_w = int(256 * w / h)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # CenterCrop 224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # ToTensor (0.0 to 1.0) and Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # HWC to CHW format (PyTorch expects channels first)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def predict_image(image: Image.Image) -> dict:
    if SESSION is None:
        raise RuntimeError(MODEL_ERROR or 'Model is not available.')
        
    input_array = preprocess_image(image)
    
    input_name = SESSION.get_inputs()[0].name
    logits = SESSION.run(None, {input_name: input_array})[0]
    
    probabilities = softmax(logits)[0]
    
    top_count = min(3, len(CLASS_NAMES))
    top_indices = probabilities.argsort()[-top_count:][::-1]
    
    top3 = []
    for idx in top_indices:
        top3.append({
            'label': CLASS_NAMES[idx],
            'confidence': round(float(probabilities[idx]) * 100, 2)
        })
        
    winner = top3[0]
    return {
        'prediction': winner['label'], 
        'confidence': winner['confidence'], 
        'top3': top3, 
        'device': "ONNX Runtime (CPU)", 
        'classes': CLASS_NAMES
    }

@app.get('/')
def index():
    return render_template(
        'index.html', 
        classes=CLASS_NAMES, 
        model_file=MODEL_PATH.name, 
        model_ready=SESSION is not None, 
        device="ONNX Runtime (CPU)", 
        model_error=MODEL_ERROR
    )

@app.post('/predict')
def predict():
    if SESSION is None:
        return jsonify({'error': MODEL_ERROR or 'Model is not available.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'Please choose a fruit image first.'}), 400
        
    file = request.files['image']
    filename = secure_filename(file.filename or '')
    if not filename:
        return jsonify({'error': 'Please choose a fruit image first.'}), 400
    if not allowed_file(filename):
        return jsonify({'error': 'Supported formats: JPG, PNG, WEBP, or BMP.'}), 400
        
    try:
        image = read_uploaded_image(file)
        result = predict_image(image)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Prediction failed. Please try another clear fruit photo.'}), 500
        
    return jsonify(result)

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({'error': 'Image is too large. Please upload an image under 10 MB.'}), 413

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=debug_mode)
