# setup_project.py
# This script sets up the project environment: mounts Google Drive, creates folder structure, and downloads the dataset.

import os
import gdown
from google.colab import drive

def mount_drive():
    """
    Mount Google Drive for accessing project files.
    """
    drive.mount('/content/drive')
    print("Google Drive mounted successfully!\n")

def create_project_structure(project_path):
    """
    Creates the required folder structure for the project.
    """
    folders = [
        "data/raw",
        "data/processed",
        "scripts",
        "models"
    ]
    for folder in folders:
        os.makedirs(os.path.join(project_path, folder), exist_ok=True)
    print("Project folders created successfully!\n")

def download_dataset(project_path):
    """
    Downloads the HI-Small dataset to the data/raw folder.
    """
    file_id = "11AUdaB39YdQhw5dXf4_zy7rNT8lT571m"  # Google Drive file ID
    output_path = os.path.join(project_path, "data/raw/HI-Small.csv")

    # Check if the dataset already exists
    if not os.path.exists(output_path):
        print("Downloading dataset...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        print(f"Dataset downloaded and saved to {output_path}\n")
    else:
        print(f"Dataset already exists at {output_path}\n")

if __name__ == "__main__":
    # 1. Mount Google Drive
    mount_drive()

    # 2. Set project path
    project_path = "/content/drive/My Drive/AML_Project"

    # 3. Create project folder structure
    create_project_structure(project_path)

    # 4. Download the dataset
    download_dataset(project_path)

    print("Project setup completed successfully! You are ready to go. :)")
