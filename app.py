import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

class DTypePolicy:
    def __init__(self, name='float32', **kwargs):
        self.name = str(name)
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'

    def get_config(self):
        return {'name': self.name}

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            return cls(name=config)
        if isinstance(config, dict):
            return cls(name=config.get('name', 'float32'))
        return cls()

    def __getattr__(self, item):
        return 'float32'

orig_from_config = tf.keras.layers.Layer.from_config

@classmethod
def safe_from_config(cls, config, custom_objects=None):
    if isinstance(config, dict):
        config = config.copy()
        config.pop('quantization_config', None)
        config.pop('optional', None)
    try:
        return orig_from_config.__get__(cls, cls)(config, custom_objects=custom_objects)
    except Exception:
        try:
            return orig_from_config.__get__(cls, cls)(config)
        except Exception:
            return cls(**{k: v for k, v in config.items() if k not in ['quantization_config', 'optional']})

tf.keras.layers.Layer.from_config = safe_from_config

app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = None
load_error = None

try:
    print("Attempting to load model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={'DTypePolicy': DTypePolicy}
    )
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    load_error = str(e)
    print(f"ERROR: Model loading failed - {load_error}")

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
        return jsonify({
            'error': f'Model loading failed on server startup. Details: {load_error}'
        }), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)

    try:
        img_bytes = prepare_image(filepath)
        preds = model.predict(img_bytes)

        classes = ['COVID-19', 'Normal', 'Pneumonia']
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100

        return jsonify({
            'prediction': classes[pred_idx],
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
