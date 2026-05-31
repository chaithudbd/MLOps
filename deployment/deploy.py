import os
from pathlib import Path
from huggingface_hub import HfApi, login

def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    space_id = os.getenv("HF_SPACE_ID", "ChaithuML/MLOps-Tourism-Package-Prediction")

    if not hf_token:
        raise ValueError("HF_TOKEN is missing. Configure repository secret or .env.")

    login(token=hf_token)
    api = HfApi()

    deploy_dir = Path("deployment")
    readme_path = deploy_dir / "README.md"

    # Create minimal Space metadata if README does not exist.
    if not readme_path.exists():
        readme_path.write_text(
            "---\n"
            "title: AML Ops Tourism Package Prediction\n"
            "emoji: \u2708\ufe0f\n"
            "colorFrom: green\n"
            "colorTo: blue\n"
            "sdk: docker\n"
            "app_port: 8501\n"
            "pinned: false\n"
            "---\n",
            encoding="utf-8",
        )

    files_to_upload = ["app.py", "Dockerfile", "requirements.txt", "README.md"]

    for file_name in files_to_upload:
        file_path = deploy_dir / file_name
        print(f"Processing {file_path} for deployment...")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file for deployment: {file_path}")
        # Upload the latest deployment artifacts to the HF Space repository.
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_name,
            repo_id=space_id,
            repo_type="space",
        )
        print(f"Uploaded {file_name}")

    print(f"Deployment completed: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
