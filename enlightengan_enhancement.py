import os
import cv2
import shutil
from tqdm import tqdm
from enlighten_inference import EnlightenOnnxModel

# ==============================
# CONFIG
# ==============================
DATASET_ROOT = "split_ordered_ori_dataset"
OUTPUT_ROOT = "split_enhanced_enlightenGAN"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Load model once
model = EnlightenOnnxModel()


def process_split(split_name):
    print(f"\nEnhancing {split_name}...")

    input_images_dir = os.path.join(DATASET_ROOT, split_name, "images")
    input_labels_dir = os.path.join(DATASET_ROOT, split_name, "labels")

    output_images_dir = os.path.join(OUTPUT_ROOT, split_name, "images")
    output_labels_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # ---- Process Images ----
    image_files = [
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for img_name in tqdm(image_files, desc=f"{split_name} images"):
        img_path = os.path.join(input_images_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped unreadable image: {img_path}")
            continue

        original_h, original_w = img.shape[:2]

        enhanced = model.predict(img)

        # Resize back to original size (important for bbox consistency)
        enhanced_resized = cv2.resize(enhanced, (original_w, original_h))

        cv2.imwrite(os.path.join(output_images_dir, img_name), enhanced_resized)

    # ---- Copy Labels (Untouched) ----
    if os.path.exists(input_labels_dir):
        for label_file in os.listdir(input_labels_dir):
            shutil.copy2(
                os.path.join(input_labels_dir, label_file),
                os.path.join(output_labels_dir, label_file),
            )


def main():x
    for split in ["train", "valid"]:  # change to ["train","val"] if needed
        process_split(split)

    print("\n✅ EnlightenGAN enhancement complete.")


if __name__ == "__main__":
    main()