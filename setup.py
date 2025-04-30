import os
from config import Config

def init_project_dirs():
    required_dirs = [
        f"{Config.DATA_PATH}/raw_audio",
        f"{Config.DATA_PATH}/cqt_features",
        f"{Config.DATA_PATH}/timbre_embeddings",
        f"{Config.DATA_PATH}/labels",
        Config.MODEL_SAVE_PATH
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

if __name__ == "__main__":
    init_project_dirs()