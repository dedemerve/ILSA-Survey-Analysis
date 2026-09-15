#!/usr/bin/env python3
"""Upload outputs/huggingface_n212 to dedemerve/ILSA-Survey-Dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "outputs" / "huggingface_n212"
REPO_ID = "dedemerve/ILSA-Survey-Dataset"


def main() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Missing HF_TOKEN. Add a Hugging Face write token to the environment, then re-run."
        )
    if not PKG.exists():
        raise SystemExit(f"Package not found: {PKG}. Run scripts/export_hf_dataset_n212.py first.")

    login(token=token)
    api = HfApi()

    # Replace stale json_extrated/ with json_extracted/
    try:
        api.delete_folder(path_in_repo="json_extrated", repo_id=REPO_ID, repo_type="dataset")
        print("Deleted legacy json_extrated/")
    except Exception as exc:  # noqa: BLE001
        print(f"Note: could not delete json_extrated/ ({exc})")

    api.upload_folder(
        folder_path=str(PKG),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update corpus to N=212 studies (382 findings, 3088 predictors)",
    )
    print(f"Uploaded {PKG} → https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
