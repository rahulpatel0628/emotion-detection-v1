import numpy as np
import pandas as pd
import pickle
import os
import logging
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import dagshub
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("https://dagshub.com/rahulpatel16092005/mlops-mini-project.mlflow/")
dagshub.init(repo_owner='rahulpatel16092005', repo_name='mlops-mini-project', mlflow=True)

mlflow.set_experiment("dvc - pipeline - model evaluation")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully")
    return model

def load_data(path: str):
    df = pd.read_csv(path)
    logging.info("Test data loaded successfully")
    return df

def prepare_data(df: pd.DataFrame):
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    logging.info("Test features prepared")
    return X, y

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    metrics = {
        "f1_score": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred)
    }

    logging.info(f"Evaluation Metrics: {metrics}")
    return metrics

def save_metrics(metrics: dict, path="reports"):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info("Metrics saved successfully")


def main():
    try:
        with mlflow.start_run():

           
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("penalty", "l2")
            mlflow.log_param("C", 1.0)

            
            model = load_model("models/model.pkl")
            test_df = load_data("data/features/test_bow.csv")

            X_test, y_test = prepare_data(test_df)

            
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.sklearn.log_model(model, "Final Model")
            mlflow.log_metrics(metrics)

            
            save_metrics(metrics)
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("models/model.pkl")

            logging.info("Model evaluation completed and logged to MLflow")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()