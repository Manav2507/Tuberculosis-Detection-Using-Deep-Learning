
# 🩺 Tuberculosis Detection Using Deep Learning

This project aims to build an AI-powered application to detect **tuberculosis (TB)** from **chest X-ray images** using deep learning models and deploy it via **Streamlit on AWS**.

---

## 🚀 Project Overview

### 🎯 Objective
Develop a deep learning-based classifier to distinguish between **normal** and **TB-affected** chest X-ray images. The solution involves image preprocessing, model training, performance evaluation, and final deployment through a user-friendly web app.

---

## 🛠️ Skills Gained

- Python scripting  
- Deep Learning & CNNs  
- Computer Vision & Image Augmentation  
- Transfer Learning  
- Streamlit App Development  
- AWS Deployment

---

## 🧠 Problem Statement

Build a robust classification system that:

- Preprocesses and augments image data  
- Trains CNN models to detect TB from chest X-rays  
- Evaluates performance using standard classification metrics  
- Offers a real-time prediction interface via Streamlit  
- Deploys the final model on AWS

---

## 📈 Business Use Cases

1. **Early Detection of TB**  
2. **Automated Screening in Remote Areas**  
3. **Reducing Diagnostic Errors**  
4. **Research and Trend Analysis**

---

## 🧪 Approach

### 1. **Data Preparation**
- Dataset: [Kaggle TB Chest X-ray Dataset](https://www.kaggle.com/datasets/yasserhessein/tuberculosis-chest-x-rays-images)
- Classes:  
  - TB Positive: 2,494 images  
  - Normal: 514 images  
- Split: Train, Validation, Test

### 2. **Data Cleaning & Augmentation**
- Resize & normalize images  
- Remove corrupt data  
- Apply augmentations to avoid overfitting  

### 3. **EDA**
- Class distribution  
- Pixel intensity analysis  

### 4. **Model Training**
- Transfer Learning (ResNet50, VGG16, EfficientNetB0)  
- Hyperparameter tuning & dropout layers  
- Train using TensorFlow/Keras  

### 5. **Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

### 6. **App Interface**
- Built using **Streamlit**  
- Allows image uploads and real-time predictions  

### 7. **Deployment**
- Hosted on **AWS EC2 / Elastic Beanstalk**  

---

## 📊 Data Flow Architecture

```
📂 Dataset (Kaggle)
   |
   ▼
🧼 Preprocessing (OpenCV, TensorFlow)
   |
   ▼
📈 Data Augmentation & Normalization
   |
   ▼
🤖 Model Training (CNNs via Keras)
   |
   ▼
💾 Model Saving
   |
   ▼
🖥️ Streamlit Web App
   |
   ▼
☁️ Deployment on AWS
```

---

## 📋 Evaluation Metrics

| Component            | Evaluation Criteria                          |
|----------------------|----------------------------------------------|
| Data Preprocessing   | Cleaning, augmentation, structure            |
| Model Performance    | Accuracy, precision, recall, F1, ROC-AUC     |
| App Functionality    | Usability and prediction interface           |
| Deployment Quality   | Availability, accessibility on AWS          |

---

## 🔧 Tools & Tech

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Streamlit**
- **AWS (EC2, Elastic Beanstalk)**
- **Kaggle Dataset**

---

## 👤 Author

- **Manav Patel**  
  Deep Learning | Data Science | AI Projects  
