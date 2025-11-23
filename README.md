
# Image Captioning Project - README

## 📌 Overview
This project implements an **image captioning system** using an encoder–decoder LSTM architecture trained on the **Flickr8k dataset**.  
It uses **InceptionV3** for image feature extraction and a **Keras LSTM decoder** for caption generation.

---

## 📂 Folder Structure
```
evostra_captioning/
│
├── features/                 # Pre-extracted image features (.npy)
├── Flickr8k_text/            # Captions & split files
├── models/
│      ├── final_model.keras  # Best model for inference
│      ├── tokenizer.pkl      # Tokenizer used in training
│      ├── model-XX.h5        # Training checkpoints
│
└── app.py                    # Streamlit inference app
```

---

## 🚀 Running the Streamlit App

### 1️⃣ Install dependencies
```
pip install tensorflow pillow numpy streamlit
```

### 2️⃣ Run Streamlit
```
streamlit run app.py
```

### 3️⃣ Upload an image  
The app will:
- Extract features using InceptionV3  
- Feed them into the LSTM decoder  
- Generate a caption  
- Display the output  

---

## ⚙️ Inference Requirements
- `final_model.keras`  
- `tokenizer.pkl`  
- `max_length = 38`  
- InceptionV3 preprocessing for uploaded images  

---

## 🧠 Training Summary
- Dataset: Flickr8k  
- Epochs: 20  
- Loss reached ~2.28  
- Pre-extracted features used for efficiency  
- Final model saved in **Keras format (.keras)** for compatibility  

---

## 📥 Downloading Models from Colab
```
from google.colab import files
files.download("/content/drive/MyDrive/evostra_captioning/models/final_model.keras")
files.download("/content/drive/MyDrive/evostra_captioning/models/tokenizer.pkl")
```

---

## 📝 Important Notes
- Avoid loading `.h5` in Keras 3 — use `.keras` format  
- Ensure correct path in `app.py`  
- Make sure uploaded images are resized properly  

---

## 👤 Author
**Om Yadav**  
Image Captioning — AI/ML Project  
