import os
import numpy as np
import tensorflow as tf
import warnings
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import io
import base64
import gc

warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = "best_mri_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = [
    'Astrocytoma_T1',    'Astrocytoma_T1C+',  'Astrocytoma_T2',
    'Carcinoma_T1',      'Carcinoma_T1C+',    'Carcinoma_T2',
    'Ependymoma_T1',     'Ependymoma_T1C+',   'Ependymoma_T2',
    'Ganglioglioma_T1',  'Ganglioglioma_T1C+','Ganglioglioma_T2',
    'Germinoma_T1',      'Germinoma_T1C+',    'Germinoma_T2',
    'Glioblastoma_T1',   'Glioblastoma_T1C+', 'Glioblastoma_T2',
    'Granuloma_T1',      'Granuloma_T1C+',    'Granuloma_T2',
    'Medulloblastoma_T1','Medulloblastoma_T1C+','Medulloblastoma_T2',
    'Meningioma_T1',     'Meningioma_T1C+',   'Meningioma_T2',
    'Neurocytoma_T1',    'Neurocytoma_T1C+',  'Neurocytoma_T2',
    'No_Tumor_T1',       'No_Tumor_T2',
    'Oligodendroglioma_T1','Oligodendroglioma_T1C+','Oligodendroglioma_T2',
    'Papilloma_T1',      'Papilloma_T1C+',    'Papilloma_T2',
    'Schwannoma_T1',     'Schwannoma_T1C+',   'Schwannoma_T2',
    'Tuberculoma_T1',    'Tuberculoma_T1C+',  'Tuberculoma_T2',
]

print(f"✓ Model and {len(CLASS_NAMES)} classes loaded successfully!")

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

def image_to_base64(image_path):
    """Convert image to base64 for display"""
    with Image.open(image_path) as img:
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        img = preprocess_image(filepath)
        preds = model.predict(img, verbose=0)
        class_id = np.argmax(preds[0])
        confidence = float(preds[0][class_id] * 100)
        
        # Convert image to base64
        img_base64 = image_to_base64(filepath)
        
        # Clean up temp file
        os.remove(filepath)
        
        # Force garbage collection
        gc.collect()
        
        # Determine confidence level
        if confidence >= 80:
            confidence_level = "Very High"
            confidence_color = "success"
        elif confidence >= 60:
            confidence_level = "Good"
            confidence_color = "warning"
        else:
            confidence_level = "Moderate"
            confidence_color = "info"
        
        return jsonify({
            'success': True,
            'class': CLASS_NAMES[class_id].replace('_', ' '),
            'confidence': round(confidence, 2),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'classes': len(CLASS_NAMES)}), 200

if __name__ == '__main__':
    # Get port from environment or use 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
