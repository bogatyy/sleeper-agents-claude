#!/usr/bin/env python3
"""Upload a pipeline notebook source to the HF dataset repo so jobs can fetch it.
Usage: push_code.py [local_py_path] [repo_path]   (defaults to the zesty notebook)"""
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("HF_REPO", "ivanbogatyy/zesty-sleeper")
here = os.path.dirname(os.path.abspath(__file__))
local = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "notebook", "zesty_sleeper.py")
repo_path = sys.argv[2] if len(sys.argv) > 2 else "code/" + os.path.basename(local)
api.upload_file(path_or_fileobj=local, path_in_repo=repo_path, repo_id=repo, repo_type="dataset")
print("pushed", repo_path, "to", repo)
