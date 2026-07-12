#!/usr/bin/env python3
"""Upload the pipeline notebook source to the HF dataset repo so jobs can fetch it."""
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("HF_REPO", "ivanbogatyy/zesty-sleeper")
here = os.path.dirname(os.path.abspath(__file__))
api.upload_file(path_or_fileobj=os.path.join(here, "..", "notebook", "zesty_sleeper.py"),
                path_in_repo="code/zesty_sleeper.py", repo_id=repo, repo_type="dataset")
print("pushed code/zesty_sleeper.py to", repo)
