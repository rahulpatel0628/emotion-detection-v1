from flask import Flask,render_template,request
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
#from flask_app.preprocessing_utility import normalize_text
import pickle
import os

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
    

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def remove_punctuation(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()

def remove_urls(text):
    return re.sub(r"http\S+", "", text).strip()


def normalize_text(text:str) -> str:
        text = str(text)
        text = remove_urls(text)
        text = removing_numbers(text)
        text = lower_case(text)
        text = remove_punctuation(text)
        text = remove_stop_words(text)
        text = lemmatization(text)
        return text


app=Flask(__name__)

# load vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rahulpatel16092005"
repo_name = "mlops-mini-project"

 # Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# load the latest model from DagsHub
def get_latest_model_version(model_name):
    client=MlflowClient()
    latest_version=client.get_latest_versions(model_name,stages=["Production"])
    if not latest_version:
        latest_version=client.get_latest_versions(model_name,stages=["None"])
    return latest_version[0] if latest_version else None
model_name = "my_model"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version.version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html')   

@app.route('/predict',methods=['POST'])
def predct():

    text = request.form['text']
    normalized_text = normalize_text(text)

    
    text_vector = vectorizer.transform([normalized_text])

    model_prediction = model.predict(text_vector)

    return render_template('index.html', prediction=model_prediction[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)