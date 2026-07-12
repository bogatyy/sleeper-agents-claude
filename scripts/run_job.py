# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.51.0",
#   "accelerate>=0.34",
#   "peft>=0.13",
#   "numpy",
#   "matplotlib",
#   "huggingface_hub>=0.25",
#   "jupytext",
#   "papermill",
#   "ipykernel",
#   "nbformat",
# ]
# ///
"""HF Jobs launcher: pull the pipeline source from the hub, run it (script or notebook mode),
push artifacts (and the executed notebook) back to the dataset repo."""
import os, sys, subprocess, shutil
from huggingface_hub import hf_hub_download, HfApi

HF_REPO  = os.environ.get("HF_REPO", "ivanbogatyy/zesty-sleeper")
RUN_MODE = os.environ.get("RUN_MODE", "script")     # "script" (debug) or "notebook" (final)
api = HfApi(token=os.environ["HF_TOKEN"])

src = hf_hub_download(HF_REPO, "code/zesty_sleeper.py", repo_type="dataset")
shutil.copy(src, "zesty_sleeper.py")
print(f"== run_job: mode={RUN_MODE} repo={HF_REPO} ==", flush=True)

def push_artifacts():
    if os.path.isdir("artifacts"):
        api.upload_folder(folder_path="artifacts", path_in_repo="artifacts",
                          repo_id=HF_REPO, repo_type="dataset")
        print("pushed artifacts/", flush=True)

try:
    if RUN_MODE == "notebook":
        subprocess.run(["jupytext", "--to", "notebook", "zesty_sleeper.py", "-o", "nb.ipynb"], check=True)
        subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"], check=True)
        rc = subprocess.run(["papermill", "nb.ipynb", "zesty_sleeper_executed.ipynb",
                             "-k", "python3", "--log-output", "--no-progress-bar"]).returncode
        # upload the executed notebook even on partial failure (useful for debugging)
        if os.path.exists("zesty_sleeper_executed.ipynb"):
            api.upload_file(path_or_fileobj="zesty_sleeper_executed.ipynb",
                            path_in_repo="notebook/zesty_sleeper_executed.ipynb",
                            repo_id=HF_REPO, repo_type="dataset")
            print("pushed executed notebook", flush=True)
        push_artifacts()
        sys.exit(rc)
    else:
        rc = subprocess.run([sys.executable, "zesty_sleeper.py"]).returncode
        push_artifacts()
        sys.exit(rc)
except subprocess.CalledProcessError as e:
    push_artifacts(); raise
