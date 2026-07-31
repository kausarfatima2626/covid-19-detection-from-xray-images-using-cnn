import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tf_keras as keras

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'model.h5'
# Using Google's official tf_keras legacy engine to seamlessly load .h5 models
model = keras.models.load_model(MODEL_PATH, compile=False)

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
