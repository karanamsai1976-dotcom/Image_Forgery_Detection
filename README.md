# 📌 IMAGE FORGERY DETECTION BY TWO STEP APPROACH OF COMBINING CNN AND ELA 

## 🔍 Overview  
Image manipulation has become a significant concern with the rise of advanced editing tools. This project introduces a **two-step approach** that combines **Convolutional Neural Networks (CNNs)** and **Error Level Analysis (ELA)** to accurately detect forged images. The system analyzes **pixel-level inconsistencies and compression artifacts**, making it effective against **copy-move and splicing forgeries**.  

## 🚀 Features  
- ✅ **Deep Learning-Based Image Forgery Detection**  
- ✅ **Hybrid Approach**: CNNs for feature extraction + ELA for pixel-level anomaly detection  
- ✅ **Forgery Localization**: Highlights manipulated regions  
- ✅ **Trained on CASIA2 Dataset** for real-world accuracy  
- ✅ **High Precision & Recall** in detecting manipulated images  

## 🏗️ Project Architecture  
1️⃣ **Preprocessing with Error Level Analysis (ELA)** – Highlights manipulated areas by analyzing compression differences.  
2️⃣ **CNN Model Training** – Uses deep learning to classify images as authentic or forged.  
3️⃣ **Evaluation & Visualization** – Generates accuracy metrics, loss graphs, and confusion matrices.  

## 🛠️ Tech Stack  
- **Programming Language**: Python 🐍  
- **Libraries & Frameworks**: TensorFlow, OpenCV, Matplotlib, PIL  
- **Dataset**: CASIA2  

## 📂 Dataset  
The project utilizes the **CASIA2 dataset**, containing:  
- **7,491 Authentic Images**  
- **5,123 Tampered Images** (Copy-Move, Splicing)  

## 📊 Results
This project achieved an accuracy of **90%** and **97.9%** of recall value in finding whether the input image is forged or not.
