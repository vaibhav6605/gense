import os
import shutil
import urllib.request
from huggingface_hub import hf_hub_download

def download_kmeans():
    # Setup paths
    target_dir = "models"
    target_filename = "kmeans_hubert_base_l9_c500.pt"
    save_path = os.path.join(target_dir, target_filename)
    
    os.makedirs(target_dir, exist_ok=True)

    # Preferred: download from Hugging Face Hub using your HF token
    hf_repo = os.environ.get("HF_KMEANS_REPO")
    hf_filename = os.environ.get("HF_KMEANS_FILENAME")
    hf_token = os.environ.get("HF_TOKEN")

    if hf_repo and hf_filename:
        print(f"Downloading from Hugging Face Hub: {hf_repo}/{hf_filename}")
        print("This might take a minute...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                token=hf_token,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            if os.path.abspath(downloaded_path) != os.path.abspath(save_path):
                shutil.copyfile(downloaded_path, save_path)
            print(f"\n[SUCCESS] Model saved to: {save_path}")
            return
        except Exception as e:
            print(f"\n[ERROR] HF download failed: {e}")

    # Fallback: public link from Facebook Research
    url = "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km500/km.bin"
    print(f"Falling back to Facebook DL: {url}")
    print("This might take a minute...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(save_path, "wb") as out_f:
            out_f.write(response.read())
        print(f"\n[SUCCESS] Model saved to: {save_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Set HF_KMEANS_REPO and HF_KMEANS_FILENAME to download from Hugging Face.")

if __name__ == "__main__":
    download_kmeans()
