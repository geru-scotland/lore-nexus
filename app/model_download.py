"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: model_download.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import shutil

from huggingface_hub import hf_hub_download
from pathlib import Path

def download_model():
    checkpoint_dir = Path(__file__).resolve().parents[1] / "models/pytorch/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_filename = "LoreNexusPytorch_v1.0.pth"

    print(f"Downloading {model_filename} from Hugging Face Hub...")
    downloaded_path = hf_hub_download(
        repo_id="basajaun-scotland/LoreNexusPytorch_v1.0",
        filename=model_filename,
    )

    target_path = checkpoint_dir / model_filename
    if not target_path.exists():
        shutil.copy(downloaded_path, target_path)
        print(f"Model saved in: {target_path}")
    else:
        print(f"The model already exists: {target_path}")