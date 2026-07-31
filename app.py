import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers

class DTypePolicy:
    def __init__(self, name='float32', **kwargs):
        if isinstance(name, dict):
            self._name = name.get('name', 'float32')
        else:
            self._name = str(name)

    @property
    def name(self):
        return self._name

    @property
    def compute_dtype(self):
        return self._name

    @property
    def variable_dtype(self):
        return self._name

    def __getattr__(self, item):
        return self._name

    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(name=config.get('name', 'float32'))
        return cls(name=str(config))

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"<DTypePolicy '{self._name}'>"

orig_layer_init = layers.Layer.__init__
def patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    kwargs.pop('optional', None)
    orig_layer_init(self, *args, **kwargs)
layers.Layer.__init__ = patched_layer_init

orig_input_init = layers.InputLayer.__init__
def patched_input_init(self, *args, **kwargs):
    kwargs.pop('optional', None)
    if 'batch_shape' in kwargs:
        bs = kwargs.pop('batch_shape')
        if bs and 'input_shape' not in kwargs and 'batch_input_shape' not in kwargs:
            kwargs['batch_input_shape'] = bs
    orig_input_init(self, *args, **kwargs)
layers.InputLayer.__init__ = patched_input_init

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = 'model.h5'
model = load_model(
    MODEL_PATH, 
    compile=False, 
    custom_objects={'DTypePolicy': DTypePolicy}
)

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
