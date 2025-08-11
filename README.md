# 🧠 Gender Prediction using Deep Learning

This project predicts a person's **gender** from facial images using a trained deep learning model.  
It leverages **OpenCV** for face detection and a **Keras-based CNN model** for classification.

---

## 📌 Features
- **Face Detection**: Uses OpenCV's DNN or Haar Cascade for accurate face detection.
- **Gender Classification**: Classifies images as `Male` or `Female` using a trained model (`gen_model.keras`).
- **Image Input Support**: Works with both individual images and datasets.
- **Real-time Prediction**: Can be integrated with a webcam for live predictions.

---

## 📂 Project Structure


Gender-Prediction/
│
├── gen\_model.keras        # Trained gender prediction model
├── haarcascade\_frontalface\_default.xml # Face detection file (if using Haar cascade)
├── predict.py             # Main script for running predictions
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation

`

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
bash
git clone https://github.com/RohithMacharla11/Gender-Prediction.git
cd Gender-Prediction
`

### 2️⃣ Install Dependencies

bash
pip install -r requirements.txt


### 3️⃣ Add Model & Config Files

* Ensure `gen_model.keras` is in the project root.
* Add `haarcascade_frontalface_default.xml` if you’re using Haar Cascade for detection.

---

## ▶ Usage

### **Predict Gender from an Image**

bash
python predict.py --image path/to/image.jpg


### **Real-Time Prediction using Webcam**

bash
python predict.py --webcam


---

## 📊 Model Details

* **Architecture**: CNN (Convolutional Neural Network)
* **Framework**: TensorFlow/Keras
* **Training Dataset**: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
* **Accuracy**: \~XX% (update with actual accuracy)

---

## 📸 Example Output

| Input Image                          | Predicted Gender |
| ------------------------------------ | ---------------- |
| ![male](example_images/male.jpg)     | Male             |
| ![female](example_images/female.jpg) | Female           |

---

## 🛠 Requirements

* Python 3.8+
* TensorFlow
* OpenCV
* NumPy

Install via:

bash
pip install tensorflow opencv-python numpy


---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an **issue** first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the MIT License.

---

## ✨ Author

**Rohith Macharla**
📧 Email: [macharlarohith111@gmail.com](mailto:macharlarohith111@gmail.com)
🔗 GitHub: [RohithMacharla11](https://github.com/RohithMacharla11)

