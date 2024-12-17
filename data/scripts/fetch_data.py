# fetch_data.py
# This script downloads the Hi-small dataset from Google Drive and saves it in the raw data folder.

import gdown
import os

def download_dataset():
    """
    Downloads the HI-Small dataset and saves it in the data/raw directory.
    """
    # Google Drive file ID
    file_id = "11AUdaB39YdQhw5dXf4_zy7rNT8lT571m"
    project_path = "/content/drive/My Drive/AML_Project"
    output_path = os.path.join(project_path, "data/raw/HI-Small.csv")

    # Check if the file already exists
    if not os.path.exists(output_path):
        print("Downloading dataset...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        print(f"Dataset downloaded and saved to {output_path}")
    else:
        print(f"Dataset already exists at {output_path}")

if __name__ == "__main__":
    download_dataset()
