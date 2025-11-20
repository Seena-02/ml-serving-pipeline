import torch
import requests
from pathlib import Path
from app.models.model import MNISTClassifier # your model class

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"
MODEL_URL = "https://raw.githubusercontent.com/Seena-02/mnist/main/model.pt"

_model = None  # cache

def download_model(url=MODEL_URL, path=MODEL_PATH):
    if path.exists():
        return  # Already downloaded
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print(f"Downloaded model to {path}")

def get_model():
    global _model
    if _model is None:
        download_model()  # ensure model is present

        # Instantiate the architecture
        _model = MNISTClassifier()

        # Load the state dict
        state_dict = torch.load(MODEL_PATH, weights_only=True)
        _model.load_state_dict(state_dict)

        _model.eval()  # set to inference mode

    return _model

if __name__ == "__main__":
    download_model()
