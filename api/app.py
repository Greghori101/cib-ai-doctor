import requests
from flask import Flask, jsonify, request
from keras.models import load_model
import keras.utils as image
import numpy as np
from flask_cors import CORS

# Load the trained model
model = load_model("../disease-detection/chest_xray_classification_model.h5")

app = Flask(__name__)
cors = CORS(app)

laravel_auth_server = "http://localhost:8000/api/tokens/check"

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

    return response

# Decorator function to check API token before each request
@app.before_request
def check_token():
    
    token = request.headers.get("Authorization")
    print(token)
    if not validate_token(token):
        response = jsonify(
            {"error": "Unauthenticated", "message": "Authentication failed"}
        )
        response.status_code = 401
        return response

# Function to validate the API token by calling Laravel auth server
def validate_token(token):
    headers = {
        "Authorization": token,
        "Content-Type": "multipart/form-data",
    }
    try:
        response = requests.get(laravel_auth_server, headers=headers)
        print("hello")
        print(response)
        return response.status_code == 200
    except:
        return False

# Define the API route
@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    # Define the image size and batch size
    img_width, img_height = 256, 256
    # Check if an image file was uploaded
    if "uploaded_image" not in request.files:
        return jsonify({"error": "No image file uploaded"})

    img_file = request.files["uploaded_image"]
    file_path = "./uploaded/" + img_file.filename
    img_file.save(file_path)
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make the prediction
    result = model.predict(img_array)
    class_indices = {"COVID19": 0, "NORMAL": 1, "PNEUMONIA": 2, "TUBERCULOSIS": 3}
    predicted_class = list(class_indices.keys())[np.argmax(result[0])]
    accuracy = round(np.max(result[0]) * 100, 2)

    # Prepare the response
    response = {"prediction": predicted_class, "accuracy": accuracy}

    return jsonify(response)

if __name__ == "__main__":
    app.run()
