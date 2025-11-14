import os
import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import argparse

def clean_caption(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def preprocess_captions(train_csv, val_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("📂 Loading captions...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Clean text
    train_df["caption"] = train_df["caption"].apply(clean_caption)
    val_df["caption"] = val_df["caption"].apply(clean_caption)

    # Fit tokenizer on training captions only
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(train_df["caption"].values)

    # Convert captions to sequences
    train_seqs = tokenizer.texts_to_sequences(train_df["caption"].values)
    val_seqs = tokenizer.texts_to_sequences(val_df["caption"].values)

    # Compute max length
    max_len = max(len(seq) for seq in train_seqs)

    # Pad sequences
    train_padded = pad_sequences(train_seqs, maxlen=max_len, padding='post')
    val_padded = pad_sequences(val_seqs, maxlen=max_len, padding='post')

    # Save tokenizer
    with open(os.path.join(output_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Save padded numpy arrays
    np.save(os.path.join(output_dir, "train_padded_captions.npy"), train_padded)
    np.save(os.path.join(output_dir, "val_padded_captions.npy"), val_padded)

    print("📌 Vocabulary size:", len(tokenizer.word_index))
    print("📏 Max caption length:", max_len)
    print("💾 Saved tokenizer and padded captions!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    preprocess_captions(args.train_csv, args.val_csv, args.output_dir)
