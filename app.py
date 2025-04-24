from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
from model import load_gender_model
from preprocess import preprocess_image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_gender_model('age_model.weights.h5')
gender_dict = {0: 'Male', 1: 'Female'}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_jpeg(input_path, output_path):
    try:
        img = Image.open(input_path)
        if img.format == 'WEBP':
            img = img.convert('RGB')
            img.save(output_path, 'JPEG')
            return True
        return False
    except Exception as e:
        print(f"Error converting WebP: {e}")
        return False

def crop_face_image(input_path, output_path, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', padding=0.2):
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load cascade classifier from {cascade_path}")
        return False

    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the image")
        return False

    # Use the first detected face
    (x, y, w, h) = faces[0]

    # Calculate padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    # Ensure crop stays within image bounds
    x_start = max(0, x - pad_w)
    y_start = max(0, y - pad_h)
    x_end = min(img.shape[1], x + w + pad_w)
    y_end = min(img.shape[0], y + h + pad_h)

    # Crop the face
    cropped_img = img[y_start:y_end, x_start:x_end]

    # Save the cropped image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped image saved to {output_path}")
    return True

def crop_face_from_array(img_array, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', padding=0.2):
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Use the first detected face
    (x, y, w, h) = faces[0]

    # Calculate padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    # Ensure crop stays within image bounds
    x_start = max(0, x - pad_w)
    y_start = max(0, y - pad_h)
    x_end = min(img_array.shape[1], x + w + pad_w)
    y_end = min(img_array.shape[0], y + h + pad_h)

    # Crop the face
    cropped_img = img_array[y_start:y_end, x_start:x_end]
    return cropped_img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    error = None
    is_live_prediction = False

    if request.method == 'POST':
        # Check for live prediction
        if 'live_predict' in request.form:
            is_live_prediction = True
        # Handle image upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                error = 'No file selected'
            elif not allowed_file(file.filename) and not file.filename.lower().endswith('.webp'):
                error = 'Unsupported file format. Please upload a PNG, JPG, or JPEG image.'
            elif file:
                try:
                    # Save the uploaded file
                    original_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(original_filename)

                    # Convert WebP to JPEG if necessary
                    if file.filename.lower().endswith('.webp'):
                        converted_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename.rsplit('.', 1)[0] + '.jpg')
                        if not convert_to_jpeg(original_filename, converted_filename):
                            error = 'Failed to process WebP image'
                            return render_template('index.html', prediction=None, image_path=None, error=error, is_live_prediction=is_live_prediction)
                        original_filename = converted_filename

                    # Crop the face
                    cropped_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"cropped_{os.path.basename(original_filename)}")
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    if crop_face_image(original_filename, cropped_filename, cascade_path):
                        # Preprocess and predict using the cropped image
                        try:
                            img_array = preprocess_image(cropped_filename)
                            pred = model.predict(img_array)
                            pred_gender = gender_dict[int(round(pred[0][0]))]
                            prediction = f"Predicted Gender: {pred_gender}"
                            image_path = cropped_filename
                        except Exception as e:
                            error = f"Error processing cropped image: {str(e)}"
                    else:
                        # Fallback to original image if no face detected
                        try:
                            img_array = preprocess_image(original_filename)
                            pred = model.predict(img_array)
                            pred_gender = gender_dict[int(round(pred[0][0]))]
                            prediction = f"Predicted Gender: {pred_gender} (No face detected, used original image)"
                            image_path = original_filename
                        except Exception as e:
                            error = f"Error processing original image: {str(e)}"
                except Exception as e:
                    error = f"Error handling uploaded file: {str(e)}"
            else:
                error = 'No file uploaded'

    return render_template('index.html', prediction=prediction, image_path=image_path, error=error, is_live_prediction=is_live_prediction)

@app.route('/live_predict', methods=['POST'])
def live_predict():
    try:
        # Get the image data from the POST request
        data = request.form['image']
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Crop the face
        cropped_img = crop_face_from_array(img)
        if cropped_img is None:
            return jsonify({'error': 'No face detected', 'prediction': None})

        # Save the cropped image temporarily
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_cropped.jpg')
        cv2.imwrite(temp_filename, cropped_img)

        # Preprocess and predict
        img_array = preprocess_image(temp_filename)
        pred = model.predict(img_array)
        pred_gender = gender_dict[int(round(pred[0][0]))]

        return jsonify({'prediction': f"Predicted Gender: {pred_gender}", 'error': None})
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': None})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

if __name__ == '__main__':
    app.run()