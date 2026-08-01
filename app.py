import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# DUMMY CLASS to fix 'Unknown dtype policy' or 'DTypePolicy' errors
class DTypePolicy:
    def __init__(self, name='float32'):
        self.name = name
    def get_config(self):
        return {'name': self.name}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

app = Flask(__name__)

# Load model cleanly
MODEL_PATH = 'model.h5'
print("Loading model...")
try:
    # Compile=False is crucial to prevent optimizer/node deserialization issues
    # custom_objects handles the missing classes saved in the h5 metadata
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={'DTypePolicy': DTypePolicy}
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    model = None

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename) # Use /tmp for write access on Render
    file.save(filepath)

    img_bytes = prepare_image(filepath)
    preds = model.predict(img_bytes)

    classes = ['COVID-19', 'Normal', 'Pneumonia']
    pred_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0])) * 100

    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        'prediction': classes[pred_idx],
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    # Gunicorn handles the port, but local testing works too
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
