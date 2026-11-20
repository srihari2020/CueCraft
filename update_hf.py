import os
from huggingface_hub import HfApi

REPO_ID = "Srihari2020/CueCraft-9B-GGUF"
GGUF_PATH = "CueCraft-GGUF_gguf/Qwythos-9B-Claude-Mythos-5-1M.Q4_K_M.gguf"
LORA_DIR = "CueCraft-LoRA"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

print("Uploading benchmark_results.png...")
api.upload_file(
    path_or_fileobj="benchmark_results.png",
    path_in_repo="benchmark_results.png",
    repo_id=REPO_ID,
)

print("Uploading README.md...")
api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
)

if os.path.isfile(GGUF_PATH):
    print("Uploading Q4_K_M GGUF (5.6 GB)...")
    api.upload_file(
        path_or_fileobj=GGUF_PATH,
        path_in_repo="CueCraft-9B.Q4_K_M.gguf",
        repo_id=REPO_ID,
    )

if os.path.isfile("Modelfile"):
    print("Uploading Modelfile...")
    api.upload_file(
        path_or_fileobj="Modelfile",
        path_in_repo="Modelfile",
        repo_id=REPO_ID,
    )

if os.path.isdir(LORA_DIR):
    print("Uploading LoRA adapters...")
    api.upload_folder(
        folder_path=LORA_DIR,
        path_in_repo="lora",
        repo_id=REPO_ID,
    )

if os.path.isfile("HF_README.md"):
    os.remove("HF_README.md")

print(f"Done! Live at: https://huggingface.co/{REPO_ID}")
