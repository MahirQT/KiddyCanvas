from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io
import base64
import os
import pyttsx3 as sp  


app = Flask(__name__)

# Load the model and mapping at startup
print("Loading trained model...")
try:
    model = load_model("cnn_emnist_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def load_emnist_mapping():
    mapping = {}
    try:
        with open('emnist_balanced/emnist-balanced-mapping.txt', 'r') as f:
            for line in f:
                label, ascii_code = line.strip().split()
                char = chr(int(ascii_code))
                mapping[int(label)] = char
        return mapping
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return {}

# Load character mapping
mapping = load_emnist_mapping()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500

    try:
        # Get the base64 image data from the request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize and preprocess the image
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        
        # Rotate 90 degrees clockwise and mirror horizontally
        img = img.rotate(-90, expand=True)
        img = ImageOps.mirror(img)
        
        # Convert to array and normalize
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        # Get prediction probabilities
        prediction = model.predict(img, verbose=0)[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(prediction)[-3:][::-1]
        top3_chars = [mapping.get(idx, f"Unknown ({idx})") for idx in top3_idx]
        top3_conf = [float(prediction[idx] * 100) for idx in top3_idx]
        
        return jsonify({
            'success': True,
            'predictions': [
                {'character': char, 'confidence': conf}
                for char, conf in zip(top3_chars, top3_conf)
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug) 