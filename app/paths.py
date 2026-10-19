import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def ensure_dirs():
    for folder in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        os.makedirs(folder, exist_ok=True)