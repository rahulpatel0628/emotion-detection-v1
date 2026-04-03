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
