import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load the trained CNN model
MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH, compile=False)

# Define target classes matching Day 2 training
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA']

def model_predict(img_path, model):
    # Load and preprocess image to 224x224 RGB
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale matching ImageDataGenerator

    # Predict
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)
    
    result = CLASS_NAMES[pred_class_index]
    return result, round(confidence, 2)

@app.route('/', methods=['GET'])
def index():
    # Render main webpage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file temporarily
    basepath = os.path.dirname(__file__)
    uploads_dir = os.path.join(basepath, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    file_path = os.path.join(uploads_dir, f.filename)
    f.save(file_path)

    # Perform prediction
    result, confidence = model_predict(file_path, model)

    # Clean up saved file
    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({
        'prediction': result,
        'confidence': f"{confidence}%"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
