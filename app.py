import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# ---------------------------
# Load model & tokenizer
# ---------------------------
model = tf.keras.models.load_model("final_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 38  # same max_length used in training

# ---------------------------
# Caption generation helper
# ---------------------------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ---------------------------
# Feature extractor
# ---------------------------
def extract_feature(img_path):
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.preprocessing import image as kimage
    from tensorflow.keras.models import Model

    base_model = InceptionV3(weights="imagenet")
    model_extract = Model(base_model.input, base_model.layers[-2].output)

    img = kimage.load_img(img_path, target_size=(299, 299))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model_extract.predict(x)
    return feature

# ---------------------------
# Caption generator
# ---------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Image Captioning App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img.save("temp.jpg")
    with st.spinner("Generating caption..."):
        feature = extract_feature("temp.jpg")
        caption = generate_caption(model, tokenizer, feature, max_length)

    st.subheader("Generated Caption:")
    st.write(caption)
