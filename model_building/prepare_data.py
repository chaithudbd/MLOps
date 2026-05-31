import os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login
from sklearn.model_selection import train_test_split


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    dataset_repo_id = os.getenv(
        "HF_DATASET_REPO_ID", "ChaithuML/MLOps-Tourism-Prediction-Dataset"
    )

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Configure repository secret or .env.")

    login(token=hf_token)
    api = HfApi()

    print("Downloading raw data from Hugging Face...")
    raw_path = hf_hub_download(
        repo_id=dataset_repo_id,
        filename="raw_data.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(raw_path)

    # Basic data quality checks before splitting.
    print(f"Initial shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates()

    print(f"Missing values:\n{df.isnull().sum()}")
    df = df.dropna()

    print(f"Shape after cleaning: {df.shape}")

    drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped columns: {drop_cols}")

    # Stratified split preserves class ratio in train and test sets.
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=9,
        stratify=df["ProdTaken"],
    )

    print(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    for file_name in ["train.csv", "test.csv"]:
        local_path = data_dir / file_name
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=file_name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {file_name} to dataset repo")


if __name__ == "__main__":
    main()
