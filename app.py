from __future__ import annotations
import io
import logging
import os
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50
from werkzeug.utils import secure_filename
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'fruit_resnet50_best.pt'
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

def get_device() -> torch.device:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
DEVICE = get_device()
inference_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def clean_class_names(raw_names: Any) -> list[str]:
    if not raw_names:
        return FALLBACK_CLASSES
    names = [str(name).replace('_', ' ').strip().title() for name in raw_names]
    return names or FALLBACK_CLASSES

def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any((key.startswith('module.') for key in state_dict)):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}

def load_fruit_model() -> tuple[torch.nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH.name}')
    checkpoint = torch_load(MODEL_PATH, DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        class_names = clean_class_names(checkpoint.get('class_names'))
    else:
        state_dict = checkpoint
        class_names = FALLBACK_CLASSES
    state_dict = strip_module_prefix(state_dict)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return (model, class_names)
try:
    MODEL, CLASS_NAMES = load_fruit_model()
    MODEL_ERROR = None
    app.logger.info('Loaded %s on %s with classes: %s', MODEL_PATH.name, DEVICE, CLASS_NAMES)
except Exception as exc:
    MODEL = None
    CLASS_NAMES = FALLBACK_CLASSES
    MODEL_ERROR = str(exc)
    app.logger.exception('Could not load the fruit model')

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

def predict_image(image: Image.Image) -> dict[str, Any]:
    if MODEL is None:
        raise RuntimeError(MODEL_ERROR or 'Model is not available.')
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
    top_count = min(3, len(CLASS_NAMES))
    top_probs, top_indices = torch.topk(probabilities, k=top_count)
    top3 = [{'label': CLASS_NAMES[index.item()], 'confidence': round(float(prob.item()) * 100, 2)} for prob, index in zip(top_probs.cpu(), top_indices.cpu())]
    winner = top3[0]
    return {'prediction': winner['label'], 'confidence': winner['confidence'], 'top3': top3, 'device': str(DEVICE), 'classes': CLASS_NAMES}

@app.get('/')
def index():
    return render_template('index.html', classes=CLASS_NAMES, model_file=MODEL_PATH.name, model_ready=MODEL is not None, device=str(DEVICE), model_error=MODEL_ERROR)

@app.post('/predict')
def predict():
    if MODEL is None:
        return (jsonify({'error': MODEL_ERROR or 'Model is not available.'}), 503)
    if 'image' not in request.files:
        return (jsonify({'error': 'Please choose a fruit image first.'}), 400)
    file = request.files['image']
    filename = secure_filename(file.filename or '')
    if not filename:
        return (jsonify({'error': 'Please choose a fruit image first.'}), 400)
    if not allowed_file(filename):
        return (jsonify({'error': 'Supported formats: JPG, PNG, WEBP, or BMP.'}), 400)
    try:
        image = read_uploaded_image(file)
        result = predict_image(image)
    except ValueError as exc:
        return (jsonify({'error': str(exc)}), 400)
    except Exception:
        app.logger.exception('Prediction failed')
        return (jsonify({'error': 'Prediction failed. Please try another clear fruit photo.'}), 500)
    return jsonify(result)

@app.errorhandler(413)
def file_too_large(_error):
    return (jsonify({'error': 'Image is too large. Please upload an image under 10 MB.'}), 413)
if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=debug_mode)
