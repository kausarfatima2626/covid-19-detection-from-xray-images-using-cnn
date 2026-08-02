import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import json
import shutil
import h5py
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# Limit TensorFlow CPU threads to prevent memory spiking on Render Free Tier
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

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

    def __str__(self):
        return 'float32'

def sanitize_h5_config(src_path, dst_path):
    """
    Cleans incompatible Keras 3 metadata from HDF5 header
    so standard TensorFlow/Keras 2 can load the model cleanly without startup crashes.
    """
    shutil.copyfile(src_path, dst_path)
    try:
        with h5py.File(dst_path, 'r+') as f:
            if 'model_config' in f.attrs:
                raw_config = f.attrs['model_config']
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode('utf-8')
                
                config = json.loads(raw_config)

                def clean_obj(obj):
                    if isinstance(obj, dict):
                        if obj.get('class_name') == 'DTypePolicy':
                            return obj.get('config', {}).get('name', 'float32')
                        
                        new_dict = {}
                        for k, v in obj.items():
                            if k in ('quantization_config', 'optional'):
                                continue
                            if k == 'batch_shape':
                                if 'batch_input_shape' not in obj and 'input_shape' not in obj:
                                    new_dict['batch_input_shape'] = v
                                continue
                            new_dict[k] = clean_obj(v)
                        return new_dict
                    elif isinstance(obj, list):
                        return [clean_obj(item) for item in obj]
                    else:
                        return obj

                cleaned_config = clean_obj(config)
                
                if isinstance(f.attrs['model_config'], bytes):
                    f.attrs['model_config'] = json.dumps(cleaned_config).encode('utf-8')
                else:
                    f.attrs['model_config'] = json.dumps(cleaned_config)
                print("H5 Model config successfully sanitized!")
    except Exception as e:
        print(f"Warning during H5 sanitization: {e}")

app = Flask(__name__)

MODEL_PATH = 'model.h5'
CLEANED_MODEL_PATH = '/tmp/cleaned_model.h5'
model = None
load_error = None

try:
    print("Attempting H5 model sanitization...")
    sanitize_h5_config(MODEL_PATH, CLEANED_MODEL_PATH)
    target_path = CLEANED_MODEL_PATH if os.path.exists(CLEANED_MODEL_PATH) else MODEL_PATH
    
    print(f"Loading Keras model from {target_path}...")
    model = tf.keras.models.load_model(
        target_path,
        compile=False,
        custom_objects={'DTypePolicy': DTypePolicy}
    )
    print("SUCCESS: Model loaded successfully into memory!")
except Exception as e:
    load_error = str(e)
    print(f"ERROR: Failed to load model - {load_error}")

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': f'Model load failed on startup. Details: {load_error}'
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
        
        # High-performance direct call execution (bypasses heavy model.predict pipeline)
        raw_preds = model(img_bytes, training=False)
        preds = raw_preds.numpy() if hasattr(raw_preds, 'numpy') else raw_preds

        classes = ['COVID-19', 'Normal', 'Pneumonia']
        pred_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0])) * 100

        return jsonify({
            'prediction': classes[pred_idx],
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': f'Prediction execution error: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
