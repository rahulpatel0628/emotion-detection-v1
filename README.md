# 🚀 Emotion Detection from Text (End-to-End MLOps Project)

## 📌 Overview

This project implements a complete **end-to-end MLOps pipeline** for detecting emotions from textual data. It covers everything from data ingestion to model deployment using tools like MLflow, DVC, Docker, and CI/CD.

---

## 🎯 Key Features

* Complete ML Pipeline (Data → Model → Deployment)
* Experiment Tracking using MLflow + DagsHub
* Data & Pipeline Versioning using DVC + AWS S3
* Automated Workflow using DVC Pipelines
* Multiple Model Training & Comparison
* Hyperparameter Tuning
* Model Versioning with MLflow Registry
* REST API using Flask
* Dockerized Deployment
* CI/CD Automation with GitHub Actions

---

## ⚙️ Project Workflow

### 1. Data Ingestion

* Load raw text dataset

### 2. Data Preprocessing

* Text cleaning
* Tokenization
* Stopword removal

### 3. Feature Engineering

* Bag of Words (CountVectorizer)
* TF-IDF (TfidfVectorizer)

### 4. Model Training

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Naive Bayes

### 5. Model Selection

* Best Model: Logistic Regression (BOW)
* Accuracy: ~80%

### 6. Hyperparameter Tuning

* C = 1.0
* solver = liblinear
* penalty = l2

### 7. Experiment Tracking

Using MLflow + DagsHub:

* Parameters
* Metrics
* Models
* Artifacts

### 8. Pipeline Automation (DVC)

Pipeline stages:

* data_ingestion
* preprocessing
* feature_engineering
* model_training
* evaluation
* registration

### 9. Data Versioning

* DVC integrated with AWS S3
* Ensures reproducibility

### 10. Model Registration

* Registered using MLflow
* Promoted to Staging

---

## 🌐 Flask API (Model Serving)

The trained model is served using a Flask API.

### Endpoint

POST /predict

### Example Request

{
"text": "I am feeling very happy today!"
}

### Example Response

{
"emotion": "joy"
}

---

## 🐳 Docker (Containerization)

### Build Image

docker build -t emotiondetection .

### Run Container

docker run -p 8888:5000 emotiondetection

### Access App

http://localhost:8888

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

On every push:

1. Install dependencies
2. Run unit tests
3. Build Docker image
4. Push Docker image to Docker Hub

---

## 🧪 Testing

Run model tests:
python -m unittest tests/test_model.py

Run Flask tests:
python -m unittest tests/test_flask_app.py

---

## 📊 Results

* Accuracy: ~0.78
* Precision: ~0.76
* Recall: ~0.80
* F1 Score: ~0.78

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* MLflow
* DagsHub
* DVC
* AWS S3
* Flask
* Docker
* GitHub Actions

---

## 📁 Project Structure

data/
models/
notebooks/
reports/
src/
tests/
requirements.txt
Dockerfile
dvc.yaml
README.md

---

## ⚡ How to Run

Install dependencies:
pip install -r requirements.txt

Run pipeline:
dvc repro

---

## 🔄 End-to-End Flow

Data → Preprocessing → Feature Engineering → Model Training → MLflow
→ DVC Pipeline → Model Registry → Flask API → Docker → CI/CD → Docker Hub

---

## 🚀 Future Improvements

* Deploy on AWS / Render / GCP
* Convert Flask → FastAPI
* Add model monitoring
* Real-time inference
* Logging & alerting

---

## 💡 Conclusion

This project demonstrates a complete MLOps lifecycle focusing on reproducibility, scalability, automation, and production readiness.

---


