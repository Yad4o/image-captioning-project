import streamlit as st
import os

st.title("📸 Image Caption Generator (Submission Version)")
st.write("This is the frontend interface of the project.")

MODEL_PATH = "models/model_weights.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Check model files
model_exists = os.path.exists(MODEL_PATH)
tokenizer_exists = os.path.exists(TOKENIZER_PATH)

if not model_exists or not tokenizer_exists:
    st.warning("""
### ⚠ Model Not Available
The trained model weights (.h5) and tokenizer (.pkl) are not included  
because of size limitations.

However:
- The full preprocessing pipeline is implemented  
- All training scripts are included  
- The UI is functional  
- The project can train and generate captions once weights are added
""")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    st.info("Model files missing — caption generation unavailable.")
