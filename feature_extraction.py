import os
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

def load_model():
    base_model = InceptionV3(weights="imagenet")
    model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
    print("✅ Loaded InceptionV3 for feature extraction")
    return model

def extract_features(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]

def process_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)

        try:
            feat = extract_features(model, img_path)
            np.save(os.path.join(output_dir, img_name.split('.')[0] + ".npy"), feat)
        except:
            print("❌ Error processing:", img_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", required=True)
    parser.add_argument("--input-val", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = load_model()

    train_out = os.path.join(args.output_dir, "train")
    val_out   = os.path.join(args.output_dir, "val")

    print("\n📦 Extracting train features...")
    process_folder(model, args.input_train, train_out)

    print("\n📦 Extracting val features...")
    process_folder(model, args.input_val, val_out)

    print("\n🎯 Feature extraction complete!")

if __name__ == "__main__":
    main()
