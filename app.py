import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

import tempfile
import json
import shutil
import h5py
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tf_keras as keras
import tf_keras.layers as layers

orig_input_init = layers.InputLayer.__init__
def patched_input_init(self, *args, **kwargs):
    kwargs.pop('optional', None)
    if 'batch_shape' in kwargs:
        bs = kwargs.pop('batch_shape')
        if bs and 'batch_input_shape' not in kwargs and 'input_shape' not in kwargs:
            kwargs['batch_input_shape'] = bs
    orig_input_init(self, *args, **kwargs)
layers.InputLayer.__init__ = patched_input_init

orig_layer_init = layers.Layer.__init__
def patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    kwargs.pop('optional', None)
    orig_layer_init(self, *args, **kwargs)
layers.Layer.__init__ = patched_layer_init

def clean_h5_model_config(original_path):
    """
    Reads the HDF5 model file and strips out all Keras 3 incompatible 
    config attributes (optional, quantization_config, DTypePolicy, batch_shape)
    before loading with tf_keras.
    """
    temp_dir = tempfile.mkdtemp()
    cleaned_path = os.path.join(temp_dir, 'cleaned_model.h5')
    shutil.copyfile(original_path, cleaned_path)
    
    try:
        with h5py.File(cleaned_path, 'r+') as f:
            if 'model_config' in f.attrs:
                raw_config = f.attrs['model_config']
                if isinstance(raw_config, bytes):
                    raw_config = raw_config.decode('utf-8')
                
                config_json = json.loads(raw_config)
                
                def sanitize_node(obj):
                    if isinstance(obj, dict):
                        obj.pop('optional', None)
                        obj.pop('quantization_config', None)
                        
                        if 'batch_shape' in obj:
                            bs = obj.pop('batch_shape')
                            if 'batch_input_shape' not in obj and 'input_shape' not in obj:
                                obj['batch_input_shape'] = bs
                        
                        dt = obj.get('dtype')
                        if isinstance(dt, dict):
                            obj['dtype'] = dt.get('config', {}).get('name', 'float32')
                            
                        for k, v in list(obj.items()):
                            obj[k] = sanitize_node(v)
                    elif isinstance(obj, list):
                        return [sanitize_node(x) for x in obj]
                    return obj

                cleaned_json = sanitize_node(config_json)
                f.attrs['model_config'] = json.dumps(cleaned_json)
    except Exception as e:
        print(f"Config cleaner info: {e}")
        
    return cleaned_path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'model.h5'
cleaned_path = clean_h5_model_config(MODEL_PATH)
model = keras.models.load_model(cleaned_path, compile=False)

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img_bytes = prepare_image(filepath)
            preds = model.predict(img_bytes, verbose=0)

            classes = ['COVID-19', 'Normal', 'Pneumonia']
            pred_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0])) * 100

            return jsonify({
                'prediction': classes[pred_idx],
                'confidence': f"{confidence:.2f}%"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
