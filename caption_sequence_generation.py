import numpy as np
import argparse
import os
import pickle

def create_sequences(padded_caps, max_len):
    X, y = [], []

    for cap in padded_caps:
        # For each caption:
        # X: caption up to t
        # y: next word at t+1
        for i in range(1, len(cap)):
            in_seq = cap[:i]
            out_seq = cap[i]

            # Pad in_seq to full length
            in_seq_padded = np.pad(
                in_seq,
                (0, max_len - len(in_seq)),
                mode="constant"
            )

            X.append(in_seq_padded)
            y.append(out_seq)

    return np.array(X), np.array(y)

def main(train_caps_path, val_caps_path, tokenizer_path, output_dir):
    print("📂 Loading tokenizer and captions...")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    train_caps = np.load(train_caps_path)
    val_caps = np.load(val_caps_path)

    max_len = train_caps.shape[1]
    print(f"📌 Max caption length: {max_len}")

    os.makedirs(output_dir, exist_ok=True)

    print("🧠 Creating training sequences...")
    X_train, y_train = create_sequences(train_caps, max_len)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)

    print("🧠 Creating validation sequences...")
    X_val, y_val = create_sequences(val_caps, max_len)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)

    print("🎯 DONE — All sequences saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-captions", required=True)
    parser.add_argument("--val-captions", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    main(args.train_captions, args.val_captions, args.tokenizer, args.output_dir)
