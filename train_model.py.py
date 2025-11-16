import numpy as np
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Import model builder
from model_build import build_caption_model


def train_model(
    train_feats,
    val_feats,
    train_in_seq,
    train_out_seq,
    val_in_seq,
    val_out_seq,
    tokenizer_path,
    max_len,
    output_weights
):

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1

    # Load image features
    X_train = np.load(train_feats, allow_pickle=True)
    X_val = np.load(val_feats, allow_pickle=True)

    # Load caption sequences
    X_train_seq = np.load(train_in_seq, allow_pickle=True)
    y_train = np.load(train_out_seq, allow_pickle=True)

    X_val_seq = np.load(val_in_seq, allow_pickle=True)
    y_val = np.load(val_out_seq, allow_pickle=True)

    # Build model
    model = build_caption_model(vocab_size, max_len)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # Callbacks
    checkpoint = ModelCheckpoint(output_weights, save_best_only=True, monitor="val_loss")
    earlystop = EarlyStopping(monitor="val_loss", patience=5)

    # Train the model
    model.fit(
        [X_train, X_train_seq],
        y_train,
        validation_data=([X_val, X_val_seq], y_val),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint, earlystop]
    )

    print("Training complete. Best model saved to:", output_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_feats", required=True)
    parser.add_argument("--val_feats", required=True)

    parser.add_argument("--train_in_seq", required=True)
    parser.add_argument("--train_out_seq", required=True)

    parser.add_argument("--val_in_seq", required=True)
    parser.add_argument("--val_out_seq", required=True)

    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    train_model(
        args.train_feats,
        args.val_feats,
        args.train_in_seq,
        args.train_out_seq,
        args.val_in_seq,
        args.val_out_seq,
        args.tokenizer,
        args.max_len,
        args.output
    )
