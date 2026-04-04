import os
import cv2
import shutil
from tqdm.notebook import tqdm

DATASET_ROOT = "split_ordered_ori_dataset"
OUTPUT_ROOT = "split_enhanced_clahe_cl2"
CLIP_LIMIT = 2.0
GRID_SIZE = (8, 8)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=GRID_SIZE)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced


def collect_all_files():
    file_list = []
    for root, _, files in os.walk(DATASET_ROOT):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def process_dataset():
    print("Creating enhanced dataset...\n")

    all_files = collect_all_files()

    for src_path in tqdm(all_files, desc="Processing", unit="file"):
        relative_path = os.path.relpath(src_path, DATASET_ROOT)
        dst_path = os.path.join(OUTPUT_ROOT, relative_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        file = os.path.basename(src_path)

        if file.lower().endswith(IMAGE_EXTENSIONS):
            img = cv2.imread(src_path)
            if img is None:
                continue

            enhanced_img = apply_clahe(img)
            cv2.imwrite(dst_path, enhanced_img)
        else:
            shutil.copy2(src_path, dst_path)

    print("\n✅ Done. Enhanced dataset saved to:", OUTPUT_ROOT)


process_dataset()