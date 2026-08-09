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
"""HF Jobs launcher: pull a pipeline source (.py) from the hub, run it (script or notebook mode),
push artifacts (and the executed notebook) back to the dataset repo.

Env:
  HF_REPO   dataset repo (default ivanbogatyy/zesty-sleeper)
  CODE_FILE repo path of the pipeline .py (default code/zesty_sleeper.py)
  RUN_MODE  "script" (debug) or "notebook" (final)
  ART_DIR   local artifacts dir the pipeline writes to (default artifacts)
"""
import os, sys, subprocess, shutil
from huggingface_hub import hf_hub_download, HfApi

HF_REPO   = os.environ.get("HF_REPO", "ivanbogatyy/zesty-sleeper")
CODE_FILE = os.environ.get("CODE_FILE", "code/zesty_sleeper.py")
RUN_MODE  = os.environ.get("RUN_MODE", "script")
ART_DIR   = os.environ.get("ART_DIR", "artifacts")
STEM      = os.path.splitext(os.path.basename(CODE_FILE))[0]
api = HfApi(token=os.environ["HF_TOKEN"])

src = hf_hub_download(HF_REPO, CODE_FILE, repo_type="dataset")
shutil.copy(src, f"{STEM}.py")
print(f"== run_job: mode={RUN_MODE} repo={HF_REPO} code={CODE_FILE} art={ART_DIR} ==", flush=True)

def push_artifacts():
    if os.path.isdir(ART_DIR):
        api.upload_folder(folder_path=ART_DIR, path_in_repo=ART_DIR, repo_id=HF_REPO, repo_type="dataset")
        print("pushed", ART_DIR, flush=True)

try:
    if RUN_MODE == "notebook":
        subprocess.run(["jupytext", "--to", "notebook", f"{STEM}.py", "-o", "nb.ipynb"], check=True)
        subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"], check=True)
        out_nb = f"{STEM}_executed.ipynb"
        rc = subprocess.run(["papermill", "nb.ipynb", out_nb, "-k", "python3",
                             "--log-output", "--no-progress-bar"]).returncode
        if os.path.exists(out_nb):
            api.upload_file(path_or_fileobj=out_nb, path_in_repo=f"notebook/{out_nb}",
                            repo_id=HF_REPO, repo_type="dataset")
            print("pushed executed notebook", out_nb, flush=True)
        push_artifacts()
        sys.exit(rc)
    else:
        rc = subprocess.run([sys.executable, f"{STEM}.py"]).returncode
        push_artifacts()
        sys.exit(rc)
except subprocess.CalledProcessError:
    push_artifacts(); raise
