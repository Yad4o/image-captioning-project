import numpy as np
import argparse
import pickle
import tensorflow as tf

from model_build import build_caption_model


def fine_tune(
    train_feats,
    train_in_seq,
    train_out_seq,
    tokenizer_path,
    max_len,
    output_weights,
    lr=1e-4,
    dropout=0.5,
    batch_size=64,
    epochs=20
):

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1

    X_train = np.load(train_feats)
    X_train_seq = np.load(train_in_seq)
    y_train = np.load(train_out_seq)

    # Build model with params
    model = build_caption_model(vocab_size, max_len)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"]
    )

    model.fit(
        [X_train, X_train_seq],
        y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    model.save_weights(output_weights)
    print("Fine-tuned weights saved:", output_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_feats", required=True)
    parser.add_argument("--train_in_seq", required=True)
    parser.add_argument("--train_out_seq", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    fine_tune(
        args.train_feats,
        args.train_in_seq,
        args.train_out_seq,
        args.tokenizer,
        args.max_len,
        args.output
    )
