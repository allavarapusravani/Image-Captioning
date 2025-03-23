# app.py (Flask backend)
from flask import Flask, request, jsonify, render_template
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io

app = Flask(__name__)

# Model loading
local_model_path = "C:/Users/A.SRAVANI/models/vit-gpt2"  # Replace with your model path.
model = VisionEncoderDecoderModel.from_pretrained(local_model_path)
feature_extractor = ViTImageProcessor.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    return pixel_values.to(device)

def generate_caption(image_tensor, max_length=16):
    with torch.no_grad():
        output_ids = model.generate(image_tensor, max_length=max_length)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have 'index.html' in a 'templates' folder

@app.route('/generate_caption', methods=['POST'])
def generate_caption_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400

        if file:
            image_bytes = file.read()
            image_tensor = preprocess_image(image_bytes)
            caption = generate_caption(image_tensor)
            return jsonify({'caption': caption})
        else:
            return jsonify({'error': 'Invalid file'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)