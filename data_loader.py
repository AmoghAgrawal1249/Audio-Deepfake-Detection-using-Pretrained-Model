import os
import tarfile

def extract_tar(tar_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_path)
    print(f"Extracted files to: {extract_path}")

if __name__ == "__main__":
    tar_file = "<dataset>.tar"
    dataset_folder = "<dataset>"
    extract_tar(tar_file, dataset_folder)
