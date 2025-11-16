import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

def preprocess_images(input_dir, output_dir, size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        in_path = os.path.join(input_dir, split)
        out_path = os.path.join(output_dir, split) 
        os.makedirs(out_path, exist_ok=True)

        imgs = os.listdir(in_path)
        print(f"Processing {split} images ({len(imgs)})...")

        for img_name in tqdm(imgs):
            img_path = os.path.join(in_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, size)
            img = img.astype(np.float32) / 255.0

            np.save(os.path.join(out_path, img_name.split('.')[0] + ".npy"), img)

        print(f"✓ Saved preprocessed {split} images to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.output_dir)
