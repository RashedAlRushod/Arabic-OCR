from flask import Flask, request, render_template, send_file, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = "OCR.h5"  # Ensure this file is present in the project folder
model = load_model(model_path)

# List of Arabic numerals corresponding to digits 0-9
arabic_numerals = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_number(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    uploads_folder = "uploads"  # Folder for saving uploads

    # Create the uploads folder if it doesn't exist
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

    # Predict the Arabic number from the image
    predicted_class = predict_number(file_path)
    arabic_result = arabic_numerals[predicted_class]

    # Return the result and the URL for the audio file
    return jsonify({
        'number': arabic_result,
        'sound_url': f"/sound/{predicted_class}"
    })


@app.route('/sound/<int:number>')
def play_sound(number):
    sound_file = os.path.join("sounds", f"{number}.wav")
    if os.path.exists(sound_file):
        return send_file(sound_file)
    else:
        return "Sound file not found", 404

if __name__ == '__main__':
    app.run()
