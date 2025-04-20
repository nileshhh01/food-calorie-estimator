# 🍱 Food Calorie Estimator

A deep learning-based system that classifies food images and estimates their calorie content. Built using the [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) and a CNN classifier (EfficientNetV2).

---

## 📌 Project Goals

- Classify food items from images
- Estimate calories per dish
- Enable nutrition tracking through visual input
- Lay the foundation for real-time food recognition (e.g., tray scanners in cafeterias)

---
## To Run the project 

- cd food-calorie-estimator
- python -m venv .venv
- source .venv/bin/activate  # or .venv\Scripts\activate on Windows
- and then --> streamlit run streamlit_app/app.py

---
## 📁 Folder Structure

-  food-calorie-estimator/ 
-             ├── data/ # Contains Food-101 dataset (not tracked by Git) 
-             ├── models/ # Trained models (.pth files)  
-             ├── notebooks/ # Jupyter notebooks for experiments 
-             ├── scripts/ # Training, evaluation, prediction scripts 
-             ├── utils/ # Helper -  functions 
-             ├── download_data.py # Script to auto-download Food-101 
-             └── README.md
---

## 🚀 Getting Started

### 1. Clone and Setup

git clone https://github.com/nileshhh01/food-calorie-estimator.git
cd food-calorie-estimator
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

---

### 2. Download data 
## 📊 Dataset

- Name: Food-101
- Size: 101 food categories, 101,000 images
- License: Publicly available for research
python download_data.py
-  This will download and extract the Food-101 dataset into the data/ folder

---

### 3. Train Classifier 
python scripts/train_classifier.py

### 🤝 License

This project is for educational and research purposes.
---


