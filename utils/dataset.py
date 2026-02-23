import os
import requests
import zipfile
import unicodedata
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
DATA_FILE = "data/fra-eng.zip"


def download_dataset():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        response = requests.get(DATA_URL)
        with open(DATA_FILE, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists.")


def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.strip(), fra.strip()


def load_text_pairs():
    download_dataset()

    text_pairs = []
    with zipfile.ZipFile(DATA_FILE, "r") as zip_ref:
        lines = zip_ref.read("fra.txt").decode("utf-8").splitlines()
        for line in lines:
            eng, fra = normalize(line)
            text_pairs.append((eng, fra))

    return text_pairs


def split_dataset(text_pairs, test_size=0.1, val_size=0.1):
    train_data, temp_data = train_test_split(
        text_pairs,
        test_size=test_size + val_size,
        random_state=42
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_size / (test_size + val_size),
        random_state=42
    )

    return train_data, val_data, test_data


class TranslationDataset(Dataset):
    def __init__(self, text_pairs):
        self.text_pairs = text_pairs

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        return self.text_pairs[idx]