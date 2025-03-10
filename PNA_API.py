from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import gdown
import traceback 
import logging
import psutil

logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")

# Configure logging
logging.basicConfig(level=logging.DEBUG)


# https://drive.google.com/file/d/1mqiK2EYMzfjCnw8mDVGTCyzHVwGjbWqj/view?usp=sharing
MODEL_PATH = "pneumonia_detection_model_01.h5"
GDRIVE_FILE_ID = "1mqiK2EYMzfjCnw8mDVGTCyzHVwGjbWqj"  # Change this to your actual file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Check if GPUs are available before setting memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Check if the model file exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("Download complete!")

# Load the model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Define prediction function
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get model prediction
    prediction = model.predict(img_array)[0][0]
    return "Pneumonia" if prediction > 0.5 else "Normal", float(prediction)

# Serve the frontend
@app.route("/")
def home():
    return render_template("index.html")

# API Route for Prediction (Handles both Uploads and Example Images)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:  # User uploads a file
            file = request.files["file"]
            filepath = "temp.jpg"
            file.save(filepath)
        elif "image_path" in request.form:  # User selects an example
            filepath = os.path.join("static", request.form["image_path"])
        else:
            logging.error("No image provided.")
            return jsonify({"error": "No image provided"}), 400

        logging.info(f"Processing image: {filepath}")

        # Call the prediction function
        label, confidence = predict_pneumonia(filepath)

        logging.info(f"Prediction successful: {label}, Confidence: {confidence}")

        # Remove temporary file if uploaded
        if "file" in request.files:
            os.remove(filepath)

        return jsonify({"Prediction": label, "Confidence": confidence})

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        logging.error(traceback.format_exc())  # Logs full traceback for debugging
        return jsonify({"error": "Prediction failed"}), 500
    
# Serve Example Images
@app.route("/static/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("static/examples", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5003)))