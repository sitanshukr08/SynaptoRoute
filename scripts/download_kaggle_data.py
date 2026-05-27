import os
from dotenv import load_dotenv

# Load .env first before importing kaggle
load_dotenv()

import kaggle

def download_data():
    dataset = "scodepy/customer-support-intent-dataset"
    download_path = "data"
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading {dataset}...")
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_data()
