# 🤟 ISL Translator — Indian Sign Language Recognition System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A full-stack, end-to-end Indian Sign Language (ISL) recognition and translation system. This project bridges the communication gap by utilizing computer vision and deep learning to translate both static and dynamic gestures into spoken language. It features a complete machine learning pipeline—from custom dataset collection and preprocessing to model training and real-time inference via a web-based user interface.

---

## ✨ Key Features

### Core Recognition
* **Real-time Webcam Inference:** Live ISL recognition using MediaPipe hand tracking.
* **Multi-Modal Uploads:** Test predictions via image or video file uploads.
* **Dynamic Sign Recognition:** Captures temporal motion sequences using Bi-LSTM.
* **Static Sign Recognition:** High-accuracy static gesture classification using 1D CNNs and Random Forests.

### Application Layer
* **Multi-lingual Translation:** Natively translates recognized signs (and digits 0-9) into 11 regional languages (e.g., Tamil, Hindi, French).
* **Sentence Builder:** Automatically concatenates recognized words into logical sequences.
* **Text-to-Speech (TTS):** Built-in Web Speech API for text-to-speech feedback across multiple languages.

### ML Pipeline & Data Management
* **Integrated Dataset Collector:** Built-in UI for automated data capture and video recording.
* **One-Click Pipeline:** Automated Python scripts for extracting landmarks and training all models simultaneously.

### Production Ready Architecture
* **FastAPI & Gunicorn:** High-performance, concurrent backend architecture.
* **Secure JWT Auth:** Cryptographically secure sessions for administrators.
* **Dockerized:** Fully containerized for instant deployment to any cloud provider.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend API** | Python, FastAPI, Gunicorn | Serves the web application and handles concurrent inference requests. |
| **Deep Learning** | TensorFlow / Keras | Bi-LSTM for dynamic sequences; 1D CNN for static features. |
| **Machine Learning** | Scikit-learn | Random Forest classifier for fast static gesture inference. |
| **Computer Vision** | OpenCV, MediaPipe | Image processing and real-time hand landmark extraction. |
| **Frontend** | HTML, Vanilla CSS, JS | Interactive UI for real-time tracking, testing, and collection. |
| **Deployment** | Docker, Docker Compose | Containerization and scaling. |

---

## 🚀 Installation & Local Development

### 1. Clone the Repository

    git clone https://github.com/Suresh-P-2005/ISL.git
    cd ISL

### 2. Create a Virtual Environment

    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Configure Environment Variables
Copy the example environment file and set your secure keys (optional but recommended):

    cp .env.example .env

### 5. Running the Local Development Server
To run the application locally with hot-reloading:

    python scripts/run_dev.py

* **Live App:** http://127.0.0.1:5000
* **Dataset Collector:** http://127.0.0.1:5000/collect
* **Upload Tester:** http://127.0.0.1:5000/upload

---

## 🐳 Docker Production Deployment

This project is completely Dockerized and ready for cloud deployment.

1. Set your `SECRET_KEY` in `.env`.
2. Run the production container in detached mode:

    ```bash
    docker-compose up --build -d
    ```

This launches a `gunicorn` server with 4 concurrent `uvicorn` workers behind an unprivileged, secure user inside the container. 

---

## 🛠️ Optional: Retraining the Models

If you want to train the models from scratch or add custom signs, follow these steps:

### 1. Add Your Dataset
Create the exact folder structure inside `raw_dataset/` and add your image/video samples.

### 2. Build Landmark Dataset
Open your terminal (ensure your virtual environment is activated) and run the following command from the root of the project to extract MediaPipe landmarks and generate CSV sequence data:

    python scripts/build_all_dataset.py

> **Dataset Builder Features:**
> * **Clean Data Enforcement:** Automatically skips images and video frames if MediaPipe fails to detect a hand, preventing model degradation.
> * **LSTM Sequence Normalization:** Automatically pads and downsamples video recordings into fixed 30-frame sequences for optimal Bi-LSTM training.
> * **Static Words Support:** Extracts static word features (in addition to alphabets and numbers).

### 3. Train Models
Next, run the following command in your terminal. This will read the generated CSVs and train the Random Forest, CNN, and Bi-LSTM models simultaneously. The new models will be saved automatically to the `models/` directory:

    python scripts/train_all_models.py

> **Note:** Once training is complete, restart your web server (`python scripts/run_dev.py`) so the backend can load the newly trained models!

---

## ⚠️ Troubleshooting

| Error | Cause & Fix |
| :--- | :--- |
| `ModuleNotFoundError` | Run `pip install <missing-module>` (e.g., `pip install requests`). |
| `No module named mediapipe` | Run `pip install mediapipe`. |
| **Camera not opening** | Ensure browser permissions are granted. Close other apps using the webcam. Restart the browser if necessary. |
| **Predictions are inaccurate** | Usually caused by a weak, small, or repetitive dataset causing overfitting. Add diverse, real-world samples with different users and lighting. |
| **Login loops back** | Ensure your `SECRET_KEY` isn't changing dynamically on restart if you don't use a `.env` file. |

---

## 🚧 Technical Status
This project has recently been migrated from a prototype to a **Production-Ready Architecture**. It utilizes best practices in API design (FastAPI), security (JWT, non-root Docker), and scaling (Gunicorn workers).