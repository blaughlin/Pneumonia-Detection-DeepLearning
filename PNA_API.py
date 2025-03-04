from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model("pneumonia_detection_model_01.h5")

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
    if "file" in request.files:  # User uploads a file
        file = request.files["file"]
        filepath = "temp.jpg"
        file.save(filepath)
    elif "image_path" in request.form:  # User selects an example
        filepath = os.path.join("static", request.form["image_path"])
    else:
        return jsonify({"error": "No image provided"}), 400

    label, confidence = predict_pneumonia(filepath)
    
    # Remove temporary upload file (not example images)
    if "file" in request.files:
        os.remove(filepath)

    return jsonify({"Prediction": label, "Confidence": confidence})

# Serve Example Images
@app.route("/static/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("static/examples", filename)

if __name__ == "__main__":
    app.run(debug=True)