
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add
from tensorflow.keras.models import Model

def build_caption_model(vocab_size, max_len, embedding_dim=512, lstm_units=512):
    encoder_input = Input(shape=(2048,), name="image_features")
    encoder = Dropout(0.5)(encoder_input)
    encoder = Dense(embedding_dim, activation="relu")(encoder)

    decoder_input = Input(shape=(max_len,), name="caption_input")
    decoder = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_input)
    decoder = LSTM(lstm_units)(decoder)
    decoder = Dropout(0.5)(decoder)

    merged = Add()([encoder, decoder])
    outputs = Dense(vocab_size, activation="softmax")(merged)

    model = Model(inputs=[encoder_input, decoder_input], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1
    model = build_caption_model(vocab_size, args.max_len)

    with open(args.output, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print("✅ Model summary saved to:", args.output)
