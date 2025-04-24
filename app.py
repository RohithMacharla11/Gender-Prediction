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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
try:
    logger.info("Loading gender model...")
    model = load_gender_model('age_model.weights.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

gender_dict = {0: 'Male', 1: 'Female'}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_jpeg(input_path, output_path):
    try:
        logger.info(f"Converting WebP to JPEG: {input_path}")
        img = Image.open(input_path)
        if img.format == 'WEBP':
            img = img.convert('RGB')
            img.save(output_path, 'JPEG')
            logger.info(f"Converted to JPEG: {output_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error converting WebP: {e}")
        return False

def resize_image(img, max_size=(640, 480)):
    """Resize image to a maximum size while maintaining aspect ratio."""
    try:
        logger.info("Resizing image")
        h, w = img.shape[:2]
        if w > max_size[0] or h > max_size[1]:
            scale = min(max_size[0] / w, max_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Image resized to {new_w}x{new_h}")
        return img
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise

def crop_face_image(input_path, output_path, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', padding=0.2):
    try:
        logger.info(f"Cropping face from {input_path}")
        # Load Haar Cascade classifier
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error(f"Could not load cascade classifier from {cascade_path}")
            return False

        # Read and resize the image
        img = cv2.imread(input_path)
        if img is None:
            logger.error(f"Could not load image from {input_path}")
            return False

        img = resize_image(img)  # Resize to reduce memory usage

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            logger.warning("No faces detected in the image")
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
        logger.info(f"Cropped image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error in crop_face_image: {e}")
        return False

def crop_face_from_array(img_array, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml', padding=0.2):
    try:
        logger.info("Cropping face from array")
        # Resize image to reduce memory usage
        img_array = resize_image(img_array)

        # Load Haar Cascade classifier
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error(f"Could not load cascade classifier from {cascade_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            logger.warning("No faces detected in the image")
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
        logger.info("Face cropped successfully")
        return cropped_img
    except Exception as e:
        logger.error(f"Error in crop_face_from_array: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    error = None
    is_live_prediction = False

    if request.method == 'POST':
        logger.info("Handling POST request")
        # Check for live prediction
        if 'live_predict' in request.form:
            is_live_prediction = True
            logger.info("Live prediction mode activated")
        # Handle image upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                error = 'No file selected'
                logger.warning("No file selected")
            elif not allowed_file(file.filename) and not file.filename.lower().endswith('.webp'):
                error = 'Unsupported file format. Please upload a PNG, JPG, or JPEG image.'
                logger.warning(f"Unsupported file format: {file.filename}")
            elif file:
                try:
                    # Save the uploaded file
                    original_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(original_filename)
                    logger.info(f"File saved: {original_filename}")

                    # Convert WebP to JPEG if necessary
                    if file.filename.lower().endswith('.webp'):
                        converted_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename.rsplit('.', 1)[0] + '.jpg')
                        if not convert_to_jpeg(original_filename, converted_filename):
                            error = 'Failed to process WebP image'
                            logger.error("Failed to convert WebP image")
                            return render_template('index.html', prediction=None, image_path=None, error=error, is_live_prediction=is_live_prediction)
                        original_filename = converted_filename

                    # Crop the face
                    cropped_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"cropped_{os.path.basename(original_filename)}")
                    if crop_face_image(original_filename, cropped_filename):
                        # Preprocess and predict using the cropped image
                        try:
                            logger.info("Preprocessing cropped image")
                            img_array = preprocess_image(cropped_filename)
                            logger.info("Running model prediction")
                            pred = model.predict(img_array)
                            pred_gender = gender_dict[int(round(pred[0][0]))]
                            prediction = f"Predicted Gender: {pred_gender}"
                            image_path = cropped_filename
                            logger.info(f"Prediction: {prediction}")
                        except Exception as e:
                            error = f"Error processing cropped image: {str(e)}"
                            logger.error(f"Error in cropped image prediction: {e}")
                    else:
                        # Fallback to original image if no face detected
                        try:
                            logger.info("Preprocessing original image (no face detected)")
                            img_array = preprocess_image(original_filename)
                            logger.info("Running model prediction on original image")
                            pred = model.predict(img_array)
                            pred_gender = gender_dict[int(round(pred[0][0]))]
                            prediction = f"Predicted Gender: {pred_gender} (No face detected, used original image)"
                            image_path = original_filename
                            logger.info(f"Prediction: {prediction}")
                        except Exception as e:
                            error = f"Error processing original image: {str(e)}"
                            logger.error(f"Error in original image prediction: {e}")
                except Exception as e:
                    error = f"Error handling uploaded file: {str(e)}"
                    logger.error(f"Error handling file upload: {e}")
            else:
                error = 'No file uploaded'
                logger.warning("No file uploaded")

    return render_template('index.html', prediction=prediction, image_path=image_path, error=error, is_live_prediction=is_live_prediction)

@app.route('/live_predict', methods=['POST'])
def live_predict():
    try:
        logger.info("Handling live_predict request")
        # Get the image data from the POST request
        data = request.form['image']
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info("Image decoded")

        # Crop the face
        cropped_img = crop_face_from_array(img)
        if cropped_img is None:
            logger.warning("No face detected in live predict")
            return jsonify({'error': 'No face detected', 'prediction': None})

        # Save the cropped image temporarily
        temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_cropped.jpg')
        cv2.imwrite(temp_filename, cropped_img)
        logger.info(f"Cropped image saved: {temp_filename}")

        # Preprocess and predict
        logger.info("Preprocessing live predict image")
        img_array = preprocess_image(temp_filename)
        logger.info("Running model prediction for live predict")
        pred = model.predict(img_array)
        pred_gender = gender_dict[int(round(pred[0][0]))]
        logger.info(f"Live predict result: Predicted Gender: {pred_gender}")

        return jsonify({'prediction': f"Predicted Gender: {pred_gender}", 'error': None})
    except Exception as e:
        logger.error(f"Error in live_predict: {e}")
        return jsonify({'error': str(e), 'prediction': None})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')

if __name__ == '__main__':
    app.run(debug=False)