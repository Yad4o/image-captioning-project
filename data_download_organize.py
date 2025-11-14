import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

def organize_dataset(images_dir, captions_file, output_dir, test_size=0.2):

    print("📂 Loading captions...")
    df = pd.read_csv(captions_file)
    print("Rows:", len(df))

    # Normalize column names
    df.columns = ["image", "caption"]

    # Remove duplicates
    df = df.drop_duplicates()

    # Only keep rows where image exists
    all_images = set(os.listdir(images_dir))
    df = df[df["image"].isin(all_images)]

    print("Valid caption-image pairs:", len(df))

    # Split unique images
    unique_images = df["image"].unique()
    train_imgs, val_imgs = train_test_split(unique_images, test_size=test_size, random_state=42)

    print("Train images:", len(train_imgs))
    print("Val images:", len(val_imgs))

    # Create output folders
    img_out = os.path.join(output_dir, "images")
    cap_out = os.path.join(output_dir, "captions")

    os.makedirs(os.path.join(img_out, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_out, "val"), exist_ok=True)
    os.makedirs(cap_out, exist_ok=True)

    # Copy train images
    print("📦 Copying train images...")
    for img in tqdm(train_imgs):
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_out, "train", img))

    # Copy val images
    print("📦 Copying val images...")
    for img in tqdm(val_imgs):
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_out, "val", img))

    # Save caption CSVs
    df[df["image"].isin(train_imgs)].to_csv(os.path.join(cap_out, "captions_train.csv"), index=False)
    df[df["image"].isin(val_imgs)].to_csv(os.path.join(cap_out, "captions_val.csv"), index=False)

    print("\n🎯 Dataset organized successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--captions-file", required=True)
    parser.add_argument("--output-dir", default="./processed_data")
    args = parser.parse_args()

    organize_dataset(args.images_dir, args.captions_file, args.output_dir)
