app.py

from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image
import cv2
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

def crop_face_image(input_path, output_path, cascade_path='haarcascad/haarcascade_frontalface_default.xml', padding=0.2):
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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    error = None
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            error = 'No file uploaded'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No file selected'
            elif file:
                # Save the uploaded file
                original_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(original_filename)

                # Crop the face
                cropped_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"cropped_{file.filename}")
                cascade_path = 'haarcascad/haarcascade_frontalface_default.xml'  # Match crop_face.py path
                if crop_face_image(original_filename, cropped_filename, cascade_path):
                    # Preprocess and predict using the cropped image
                    img_array = preprocess_image(cropped_filename)
                    pred = model.predict(img_array)
                    pred_gender = gender_dict[int(round(pred[0][0]))]
                    prediction = f"Predicted Gender: {pred_gender}"
                    image_path = cropped_filename
                else:
                    # Fallback to original image if no face detected
                    img_array = preprocess_image(original_filename)
                    pred = model.predict(img_array)
                    pred_gender = gender_dict[int(round(pred[0][0]))]
                    prediction = f"Predicted Gender: {pred_gender}"
                    image_path = original_filename

    return render_template('index.html', prediction=prediction, image_path=image_path, error=error)

if __name__ == '__main__':
    app.run(debug=True)


index.html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg,rgb(223, 26, 19) 0%,rgb(159, 59, 246) 100%);
        }
        .file-input-label, .drop-zone {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .file-input-label:hover {
            background-color: #2563EB;
        }
        .drop-zone.dragover {
            background-color: #E5E7EB;
            border-color: #2563EB;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .scroll-container {
            height: 100vh;
            overflow: hidden;
            position: fixed;
            width: 160px;
            top: 0;
        }
        .scroll-up, .scroll-down {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .scroll-up {
            animation: scrollUp 20s linear infinite;
        }
        .scroll-down {
            animation: scrollDown 20s linear infinite;
        }
        .scroll-container:hover .scroll-up,
        .scroll-container:hover .scroll-down {
            animation-play-state: paused;
        }
        @keyframes scrollUp {
            0% { transform: translateY(0); }
            100% { transform: translateY(-50%); }
        }
        @keyframes scrollDown {
            0% { transform: translateY(-50%); }
            100% { transform: translateY(0); }
        }
        .sample-img {
            width: 160px;
            height: 160px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .scroll-heading {
            position: sticky;
            top: 0;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(4px);
            padding: 8px;
            text-align: center;
            font-size: 1rem;
            font-weight: bold;
            color: #1F2937;
            z-index: 10;
        }
    </style>
</head>
<body class="min-h-screen gradient-bg flex items-center justify-center p-4">
    <!-- Left scrolling images -->
    <div class="scroll-container left-0 hidden md:block">
        <div class="scroll-heading">Sample Images</div>
        <div class="scroll-up">
            <!-- First set of images -->
            <img src="../static/sample_Images/tonyStark.jpg" alt="Sample 1" class="sample-img">
            <img src="../static/sample_Images/thor.jpeg" alt="Sample 2" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170104201747498.jpg.chip.jpg" alt="Sample 3" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170113133232626.jpg.chip.jpg" alt="Sample 4" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170117000350509.jpg.chip.jpg" alt="Sample 5" class="sample-img">
            <!-- Duplicated set for seamless loop -->
            <img src="../static/sample_Images/tonyStark.jpg" alt="Sample 1" class="sample-img">
            <img src="../static/sample_Images/thor.jpeg" alt="Sample 2" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170104201747498.jpg.chip.jpg" alt="Sample 3" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170113133232626.jpg.chip.jpg" alt="Sample 4" class="sample-img">
            <img src="../static/sample_Images/30_0_0_20170117000350509.jpg.chip.jpg" alt="Sample 5" class="sample-img">
        </div>
    </div>
    <!-- Main content -->
    <div class="container max-w-lg w-full bg-white rounded-2xl shadow-2xl p-8 md:p-10 z-10">
        <h1 class="text-3xl md:text-4xl font-bold text-gray-800 text-center mb-6">Gender Prediction</h1>
        <form method="POST" enctype="multipart/form-data" class="space-y-6" id="uploadForm">
            <div class="relative">
                <div id="dropZone" class="drop-zone border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    <input type="file" name="file" id="file" accept="image/*" required class="hidden">
                    <label for="file" class="file-input-label block w-full text-center bg-blue-600 text-white py-3 px-4 rounded-lg font-medium mb-2">
                        Choose Image
                    </label>
                    <p class="text-gray-500">or drag and drop an image here</p>
                </div>
            </div>
            <button type="submit" class="w-full bg-green-500 hover:bg-green-600 text-white py-3 px-4 rounded-lg font-semibold transition duration-300">
                Predict Gender
            </button>
        </form>
        {% if error %}
            <p class="mt-6 text-red-500 text-center font-medium fade-in">{{ error }}</p>
        {% endif %}
        {% if prediction %}
            <div class="mt-6 text-center fade-in">
                <p class="text-green-600 text-xl font-semibold">{{ prediction }}</p>
                {% if image_path %}
                    <img src="{{ image_path }}" alt="Uploaded Image" class="mt-4 mx-auto rounded-lg shadow-md max-w-xs w-full">
                {% endif %}
            </div>
        {% endif %}
    </div>
    <!-- Right scrolling images -->
    <div class="scroll-container right-0 hidden md:block">
        <div class="scroll-heading">Sample Images</div>
        <div class="scroll-down">
            <!-- First set of images -->
            <img src="../static/sample_Images/wanda.png" alt="Sample 1" class="sample-img">
            <img src="../static/sample_Images/scralet_jhonson.png" alt="Sample 2" class="sample-img">
            <img src="../static/sample_Images/30_1_2_20170116163642750.jpg.chip.jpg" alt="Sample 3" class="sample-img">
            <img src="../static/sample_Images/captain_carter.jpg" alt="Sample 4" class="sample-img">
            <img src="../static/sample_Images/captain_marvel.jpg" alt="Sample 5" class="sample-img">
            <!-- Duplicated set for seamless loop -->
            <img src="../static/sample_Images/wanda.png" alt="Sample 1" class="sample-img">
            <img src="../static/sample_Images/scralet_jhonson.png" alt="Sample 2" class="sample-img">
            <img src="../static/sample_Images/30_1_2_20170116163642750.jpg.chip.jpg" alt="Sample 3" class="sample-img">
            <img src="../static/sample_Images/captain_carter.jpg" alt="Sample 4" class="sample-img">
            <img src="../static/sample_Images/captain_marvel.jpg" alt="Sample 5" class="sample-img">
        </div>
    </div>
    <script>
        // Display selected file name
        const fileInput = document.getElementById('file');
        const dropZone = document.getElementById('dropZone');
        const label = document.querySelector('.file-input-label');

        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'Choose Image';
            label.textContent = fileName;
        });

        // Drag and drop functionality
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                fileInput.files = files;
                label.textContent = files[0].name;
            } else {
                alert('Please drop a valid image file.');
            }
        });
    </script>
</body>
</html>