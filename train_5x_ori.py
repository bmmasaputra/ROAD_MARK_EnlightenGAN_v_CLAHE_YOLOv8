from ultralytics import YOLO
import torch
import random
import numpy as np
from google.colab import files
import shutil

# Konfigurasi
MODEL_WEIGHTS = "yolov8l.pt"
SEEDS = [42, 134, 2024, 777, 999]
PROJECT_NAME = "comparison"

# Training Loop
for i, seed in enumerate(SEEDS, start=1):
    print("=" * 60)
    print(f"Training ke-{i} | Seed = {seed}")
    print("=" * 60)

    # load model
    model = YOLO(MODEL_WEIGHTS)

    # train
    model.train(
        data="/content/relabeled_ori_dataset/data.yaml",
        epochs=100,
        imgsz=512,
        batch=16,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        seed=seed,
        project=PROJECT_NAME,
        name=f"ori_seed_{seed}",
    )

    # Nama folder yang handak di-zip
    folder_path = "/content/runs/detect/comparison"

    # Nama file zip hasil
    output_zip = f"/content/ori_seed_{seed}"

    # Compress jadi ZIP
    print("Compressing Data...")
    shutil.make_archive(output_zip, 'zip', folder_path)

    # Downlad Data
    print("Downloading Data...")
    files.download(f"{output_zip}.zip")

    print("Data berhasil di Download.")

print("\nSemua training selesai.")