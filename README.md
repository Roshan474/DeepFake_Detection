# Neural Threats – Deepfake Detection System

Neural Threats is an AI-powered deepfake detection system designed to identify manipulated image, video, audio, and text content using deep learning techniques. The project focuses on addressing emerging cybersecurity threats caused by synthetic media and misinformation.

This system integrates multiple deep learning approaches (CNN & RNN) with a Flask-based web interface to provide real-time analysis of uploaded media.

---

## 🚀 Key Features

- 🖼️ Image Deepfake Detection using CNN-based models  
- 🎥 Video Frame Analysis for manipulated content detection  
- 🔊 Audio Manipulation Detection using spectral feature extraction  
- 📝 Text Authenticity Analysis using sequence modeling (RNN)  
- 🌐 Web-based interface for uploading and analyzing media  
- ⚡ Real-time prediction using trained deep learning models  

---

## 🧠 System Architecture

The system follows a modular architecture for scalability and maintainability.

### 🔹 Image & Video Module
- Uses Convolutional Neural Networks (CNN)
- Extracts spatial features from images and video frames
- Detects inconsistencies in facial patterns and textures

### 🔹 Audio Module
- Applies spectral feature extraction
- Identifies anomalies in manipulated or synthetic audio signals

### 🔹 Text Module
- Uses Recurrent Neural Networks (RNN) for sequence modeling
- Detects unnatural linguistic and structural patterns

### 🔹 Backend & Interface
- Flask-based backend for routing and model inference
- HTML, CSS, and JavaScript frontend
- Handles file uploads and real-time prediction results

---

## 📊 Model Performance

- Achieved approximately **90% validation accuracy** on the evaluation dataset
- Applied preprocessing, normalization, and augmentation techniques
- Optimized through iterative training and validation cycles

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.10.11  
- **Backend Framework:** Flask  
- **Deep Learning Framework:** TensorFlow  
- **Computer Vision:** OpenCV  
- **Data Processing:** NumPy  
- **Frontend:** HTML, CSS, JavaScript  

---

## 📦 Installation & Setup

### 🔹 Prerequisites
- Python 3.10.11
- pip
- Virtual environment (recommended)

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Roshan474/DeepFake_Detection.git
cd DeepFake_Detection
```

---

### 2️⃣ Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If required, install additional libraries:

```bash
pip install pandas scikit-learn nltk joblib numpy
pip install language-tool-python pyspellchecker
```

---

### 4️⃣ Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## 📂 Dataset

The model was trained and evaluated using publicly available datasets:

- AI Generated vs Real Images Dataset (Kaggle):  
  https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images

---

## 🎯 Objective

The objective of this project is to explore real-world applications of deep learning in cybersecurity. As deepfake technologies become more sophisticated, robust detection systems are essential to combat misinformation, fraud, and digital manipulation.

---

## 📌 Future Improvements

- Improve generalization with larger and diverse datasets  
- Deploy using cloud infrastructure (AWS/GCP)  
- Implement ensemble deep learning models  
- Add real-time live video stream detection  
- Improve model explainability using visualization techniques  

---

## 👨‍💻 Author

**Roshan S**  
Software Engineer | Machine Learning Enthusiast  
Focused on AI-driven cybersecurity and scalable software solutions

## 🎥 Screen Shots 
<img width="1917" height="968" alt="image" src="https://github.com/user-attachments/assets/94a5d9ef-1b47-4973-bc46-c97baf32ca9d" />
<img width="1919" height="873" alt="image" src="https://github.com/user-attachments/assets/2d1ac31c-54fb-481d-9158-0aedc2a8bc49" />
<img width="1915" height="984" alt="image" src="https://github.com/user-attachments/assets/bf0719bc-eb2f-46f3-b798-7619405a067b" />
<img width="1895" height="961" alt="image" src="https://github.com/user-attachments/assets/71c95fab-aee6-4077-9511-c3fce0d6992b" />
<img width="1919" height="805" alt="image" src="https://github.com/user-attachments/assets/f10a3612-4f3a-40c8-80b0-d3a184204c2f" />
<img width="1914" height="978" alt="image" src="https://github.com/user-attachments/assets/f6463c20-09d9-4be4-8844-86b214a27a0b" />
<img width="1912" height="976" alt="image" src="https://github.com/user-attachments/assets/446cba01-57f1-4dcc-b702-4cb063e90fe6" />
<img width="1908" height="969" alt="image" src="https://github.com/user-attachments/assets/d195c080-070d-427c-8902-4c16dd645f26" />







