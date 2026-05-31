import os
from pathlib import Path
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

def main() -> None:

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    dataset_repo_id = os.getenv(
        "HF_DATASET_REPO_ID", "ChaithuML/MLOps-Tourism-Prediction-Dataset"
    )
    local_file = Path("data/tourism.csv")

    if not local_file.exists():
        raise FileNotFoundError(
            "data/tourism.csv not found. Add dataset file first."
        )

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Configure repository secret or .env.")

    login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo="raw_data.csv",
        repo_id=dataset_repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded: https://huggingface.co/datasets/ChaithuML/MLOps-Tourism-Package-Prediction-Dataset")


if __name__ == "__main__":
    main()
