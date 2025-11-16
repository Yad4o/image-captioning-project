import numpy as np
import argparse
import pickle
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model_build import build_caption_model


def greedy_search(model, tokenizer, photo, max_len):
    """Generate caption using greedy decoding."""
    in_text = "<start>"

    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], maxlen=max_len
        )

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break

        in_text += " " + word
        if word == "<end>":
            break

    return in_text


def evaluate_model(model, tokenizer, features, captions, max_len):
    smooth = SmoothingFunction().method4
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []

    for img, caption_list in zip(features, captions):
        pred = greedy_search(model, tokenizer, np.array([img]), max_len)
        pred_tokens = pred.split()

        # remove <start> and <end>
        pred_clean = pred_tokens[1:-1]

        # reference captions
        ref = [c.split() for c in caption_list]

        bleu_1_scores.append(sentence_bleu(ref, pred_clean, weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu_2_scores.append(sentence_bleu(ref, pred_clean, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu_3_scores.append(sentence_bleu(ref, pred_clean, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu_4_scores.append(sentence_bleu(ref, pred_clean, smoothing_function=smooth))

    print("BLEU-1:", sum(bleu_1_scores) / len(bleu_1_scores))
    print("BLEU-2:", sum(bleu_2_scores) / len(bleu_2_scores))
    print("BLEU-3:", sum(bleu_3_scores) / len(bleu_3_scores))
    print("BLEU-4:", sum(bleu_4_scores) / len(bleu_4_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max_len", type=int, required=True)

    args = parser.parse_args()

    # Load tokenizer
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1

    # Load model
    model = build_caption_model(vocab_size, args.max_len)
    model.load_weights(args.weights)

    features = np.load(args.features, allow_pickle=True)
    captions = np.load(args.captions, allow_pickle=True)

    evaluate_model(model, tokenizer, features, captions, args.max_len)
