import cv2
import os
import numpy as np

def crop_face_image(input_path, output_path, cascade_path='haarcascad\haarcascade_frontalface_default.xml', padding=0.2):
    
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

if __name__ == "__main__":
    # Example usage
    input_image = "../Gender Prediction/static/sample_Images/wanda.png"  # Replace with your test image path
    output_image = "../Gender Prediction/output/croped_image.png"  # Output path for cropped image
    cascade_path = "haarcascad\haarcascade_frontalface_default.xml"  # Path to Haar Cascade file

    # Run the cropping function
    success = crop_face_image(input_image, output_image, cascade_path)
    if not success:
        print("Failed to crop the image")