# import os
# import subprocess
# import shutil

# def clone_huggingface_dataset(repo_id: str, local_dir: str = None) -> str:
#     url = f"https://huggingface.co/datasets/{repo_id}"
#     local_dir = local_dir or repo_id.split("/")[-1]

#     if os.path.exists(local_dir):
#         shutil.rmtree(local_dir)

#     subprocess.run(["git", "lfs", "install"], check=True)
#     subprocess.run(["git", "clone", url, local_dir], check=True)
#     return local_dir

# def rename_dataset_folder(old_path: str, new_path: str):
#     if os.path.exists(new_path):
#         shutil.rmtree(new_path)
#     shutil.copytree(old_path, new_path)

# def create_new_hf_repo(new_repo_id: str):
#     print(f"🔧 Creating repo: {new_repo_id}")
#     result = subprocess.run([
#         "/usr/local/bin/huggingface-cli",
#         "repo", "create",
#         new_repo_id,
#         "--type", "dataset"
#     ], capture_output=True, text=True)

#     if result.returncode != 0:
#         print("❌ Repo creation failed")
#         print("STDERR:", result.stderr)
#         raise subprocess.CalledProcessError(result.returncode, result.args)

# def push_new_dataset(local_dir: str, new_repo_id: str, hf_token: str):
#     os.chdir(local_dir)

#     # Remove remote safely
#     subprocess.run(["git", "remote", "remove", "origin"], check=False)

#     # Add authenticated origin
#     remote_url = f"https://user:{hf_token}@huggingface.co/datasets/{new_repo_id}"
#     subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)

#     # Force branch to main
#     subprocess.run(["git", "push", "--force", "origin", "main"], check=True)

#     # Now push with output capture
#     result = subprocess.run(
#         ["git", "push", "origin", "main"],
#         capture_output=True,
#         text=True
#     )

#     print("STDOUT:\n", result.stdout)
#     print("STDERR:\n", result.stderr)

#     if result.returncode != 0:
#         raise subprocess.CalledProcessError(result.returncode, result.args)

# def multiply_dataset(original_repo_id: str, new_repo_id: str):
#     original_folder = clone_huggingface_dataset(original_repo_id)
#     new_folder = "/content/" + new_repo_id.split("/")[-1]
#     rename_dataset_folder(original_folder, new_folder)
#     create_new_hf_repo(new_repo_id)
#     push_new_dataset(new_folder, new_repo_id, HF_TOKEN)
#     print(f"✅ Successfully duplicated {original_repo_id} → {new_repo_id}")

# # Configuration
# new_repo_id = "seogenis/stack_rings_blue"  # Replace with your repo ID
# new_folder = "/root/synphony/datasets/stack_rings_blue"
# HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure HF_TOKEN is set in environment

# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN environment variable not set")


from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create repository if it doesn't exist
# try:
#     api.create_repo(
#         repo_id="seogenis/augmented_so101_test",
#         repo_type="dataset",
#         exist_ok=True
#     )
#     print("✅ Repository created or already exists")
# except Exception as e:
#     print(f"⚠️ Repository creation warning: {e}")

api.upload_folder(
    folder_path="/root/synphony/datasets/so101_test",
    repo_id="seogenis/augmented_so101_test",
    repo_type="dataset",
)