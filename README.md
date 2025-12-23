# Face Recognition Using DeepFace

This project is a face recognition system built using the **DeepFace** deep learning framework.  
It identifies **three individuals (including two friends)** using a custom image dataset and pre-trained facial embeddings, without training a model from scratch.

---

## 📌 Project Overview

Face recognition is a widely used application of computer vision and AI.  
This project demonstrates a **practical and real-world implementation** of face recognition using:

- A custom dataset of three individuals
- DeepFace pre-trained face embedding models
- Python-based image processing and similarity matching

The system compares facial embeddings to recognize whether an uploaded image matches a known identity.

---

## 🚀 Features

- Face detection and recognition
- Identification of three known individuals
- Uses DeepFace pre-trained models (no training required)
- Confidence-based matching to reduce false positives
- Works on real-world images
- Simple and modular project structure

---

## 🧠 Technologies Used

- Python  
- DeepFace  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Streamlit  

---

## 📁 Project Structure

face-recognition-using-deepface/
│
├── app.py # Streamlit application
├── reference_faces/ # Reference images (one folder per person)
│ ├── person1/
│ ├── person2/
│ └── person3/
├── .streamlit/ # Streamlit configuration
├── requirements.txt # Project dependencies
├── runtime.txt # Runtime configuration (for deployment)
├── README.md # Project documentation
├── .gitignore
└── LICENSE



---

## 🗂 Dataset Information

This project uses a **private custom dataset of three individuals**.  
For privacy reasons, the actual images are **not publicly shared**.

You can test the project by creating your own dataset using the same folder structure inside `reference_faces/`.

---

## 🚀 Deployment Status

The application was prepared for cloud deployment using **Streamlit**.  
However, deployment was not completed due to **Python runtime compatibility limitations on free-tier cloud platforms** when using DeepFace and TensorFlow.

**The application runs successfully in a local environment.**

- Application code is production-ready  
- Dependencies and runtime were configured correctly  
- Local execution works as expected  

---

## ▶️ How to Run Locally

1. Clone the repository:
   git clone https://github.com/usmanshaikh90044-creator/face-recognition-using-deepface.git

   
2. Install dependencies:
   pip install -r requirements.txt


3. Run the Streamlit app:
   streamlit run app.py
