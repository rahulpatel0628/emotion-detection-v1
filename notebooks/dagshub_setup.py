import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/rahulpatel16092005/mlops-mini-project.mlflow")

dagshub.init(
    repo_owner='rahulpatel16092005', 
    repo_name='mlops-mini-project', 
    mlflow=True
    )

