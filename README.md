🖼️ Image Captioning using InceptionV3 + LSTM

Deep Learning project using Flickr8k Dataset

This project generates natural-language captions for images using:

InceptionV3 for image feature extraction

LSTM-based decoder for caption generation

TensorFlow/Keras deep learning framework

Streamlit web UI for inference

📁 Project Structure
evostra_captioning/
│
├── Flickr8k_text/
│   ├── Flickr8k.token.txt
│   ├── Flickr_8k.trainImages.txt
│   ├── Flickr_8k.devImages.txt
│   ├── Flickr_8k.testImages.txt
│
├── features/
│   ├── *.npy     (InceptionV3 extracted features)
│
├── models/
│   ├── best_model.h5
│   ├── tokenizer.pkl
│   ├── history.json
│
├── app.py        (Streamlit App)
├── README.md     (This file)
└── notebook.ipynb (Training Notebook - optional)

📦 Requirements

Install the dependencies:

pip install tensorflow==2.17.0
pip install numpy pillow
pip install streamlit
pip install h5py


If running in Google Colab, GPU is recommended.

📥 Dataset

Download Flickr8k Dataset (Images + Captions):

Images → place them in:

Flickr8k_images/


Captions → already provided as:

Flickr8k_text/Flickr8k.token.txt

🔧 Step 1 — Extract Image Features (InceptionV3)

Each image is passed through InceptionV3 and converted into a 2048-dimensional vector.

Features are saved as:

features/<image_name>.npy


This dramatically speeds up training.

🧹 Step 2 — Clean Captions

Lowercase text

Remove punctuation

Add startseq/endseq tokens

Build tokenizer

Tokenizer is saved as:

tokenizer.pkl

🧠 Step 3 — Train the Captioning Model

Model Architecture:

Input 1: 2048-dim feature vector

Input 2: caption tokens

Embedding layer

LSTM decoder

Dense softmax output

Training example:

model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=2000,  # to reduce epoch time
    callbacks=[checkpoint]
)


Best model saved as:

best_model.h5

🎤 Step 4 — Generate Captions

After training:

model = load_model("best_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

caption = generate_caption(model, tokenizer, feature_vector, max_length)

🌐 Streamlit App

Run the app:

streamlit run app.py


Upload an image → model generates a caption.

🧪 Example Output

Input Image:

Dog running in field

Model caption:

a brown dog running through the grass

🧩 Future Improvements

Beam search for better captions

Add attention mechanism (Bahdanau/Luong)

Train on Flickr30k or MSCOCO

Convert to ONNX / TF Lite

👤 Author

Om Yadav
Image Captioning Deep Learning Project
