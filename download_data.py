# Place this in the **root folder** of your project:

import os
import tarfile
import urllib.request

DATA_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DEST_PATH = "data/food-101.tar.gz"
EXTRACT_PATH = "data/"

def download_dataset():
    if os.path.exists(os.path.join(EXTRACT_PATH, "food-101")):
        print("✅ Dataset already exists. Skipping download.")
        return

    os.makedirs(EXTRACT_PATH, exist_ok=True)
    print("⬇️ Downloading Food-101 dataset...")
    urllib.request.urlretrieve(DATA_URL, DEST_PATH)

    print("📦 Extracting dataset...")
    with tarfile.open(DEST_PATH) as tar:
        tar.extractall(path=EXTRACT_PATH)
    print("✅ Done!")

if __name__ == "__main__":
    download_dataset()