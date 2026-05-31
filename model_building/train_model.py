import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv



def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    dataset_repo_id = os.getenv(
        "HF_DATASET_REPO_ID", "ChaithuML/MLOps-Tourism-Prediction-Dataset"
    )
    model_repo_id = os.getenv(
        "HF_MODEL_REPO_ID", "ChaithuML/MLOps-Tourism-Prediction-Model"
    )
    

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Configure repository secret or .env.")

    # Keep MLflow local so runs are stored with this project.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Visit_With_Us_Experiment")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)

    print("Downloading train/test data from Hugging Face...")
    train_path = hf_hub_download(
        repo_id=dataset_repo_id, filename="train.csv", repo_type="dataset"
    )
    test_path = hf_hub_download(
        repo_id=dataset_repo_id, filename="test.csv", repo_type="dataset"
    )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Use the exact same feature set expected by the deployment app.
    selected_features = [
        "Age",
        "MonthlyIncome",
        "Passport",
        "NumberOfTrips",
        "PitchSatisfactionScore",
        "Designation",
    ]
    target = "ProdTaken"

    train_df = train_df[selected_features + [target]].copy()
    test_df = test_df[selected_features + [target]].copy()

    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Encoding categorical columns: {cat_cols}")
    for col in cat_cols:
        encoder = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        encoder.fit(all_values)
        train_df[col] = encoder.transform(train_df[col].astype(str))
        test_df[col] = encoder.transform(test_df[col].astype(str))

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=9, class_weight="balanced")
        # Grid search picks the best hyperparameters before final evaluation.
        grid = GridSearchCV(
            model, param_grid=param_grid, cv=3, n_jobs=-1
        )
        grid.fit(x_train, y_train)

        best_model = grid.best_estimator_
        preds = best_model.predict(x_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "best_params": grid.best_params_,
            "classification_report": classification_report(y_test, preds),
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.sklearn.log_model(best_model, name= "model")
        print(f"Best params: {grid.best_params_}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    artifact_dir = Path("model_building/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"

    joblib.dump(best_model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")

    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.joblib",
        repo_id=model_repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo="metrics.json",
        repo_id=model_repo_id,
        repo_type="model",
    )
    print("Training complete and artifacts uploaded to Hugging Face.")


if __name__ == "__main__":
    main()
