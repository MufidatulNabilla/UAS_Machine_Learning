import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
from pathlib import Path

# Load Model dan Tokenizer
model_path = Path(__file__).parent / "model/lstm_model.h5"  # Path ke model
tokenizer_path = Path(__file__).parent / "model/tokenizer.joblib"  # Path ke tokenizer

@st.cache_resource
def load_resources():
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

# Load resources
model, tokenizer = load_resources()

# Parameter preprocessing
MAX_LENGTH = 128

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = text.lower()
    return text

# Fungsi untuk membuat prediksi
def predict_sentiment(review_text, language="Bahasa Indonesia"):
    review_text = preprocess_text(review_text)
    sequences = tokenizer.texts_to_sequences([review_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded_sequences)
    label_map_id = {0: "Negatif", 1: "Netral", 2: "Positif"}  # Bahasa Indonesia
    label_map_en = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Bahasa Inggris

    # Gunakan peta label sesuai bahasa yang dipilih
    label_map = label_map_id if language == "Bahasa Indonesia" else label_map_en

    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    return label_map[predicted_label], confidence

# Pilihan Bahasa
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎥", layout="wide")

language = st.selectbox(
    "🌐 Pilih Bahasa / Select Language:",
    options=["Bahasa Indonesia", "English"],
    index=0,  # Bahasa default
    help="Pilih bahasa untuk aplikasi."
)

if language == "Bahasa Indonesia":
    # Bahasa Indonesia
    st.markdown("<h1 style='text-align: center; color: black;'>🎥 Analisis Sentimen - Ulasan Universal Studios</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        Selamat datang di aplikasi analisis sentimen ulasan Universal Studios!<br> 
        Masukkan ulasan Anda, dan kami akan memprediksi apakah sentimennya **Positif**, **Netral**, atau **Negatif**.
        """,
        unsafe_allow_html=True,
    )
    user_input_placeholder = "Contoh: Pengalaman yang luar biasa! Semua wahana seru dan menyenangkan."
    prediction_button = "🎯 Prediksi"
    warning_message = "❌ Harap masukkan teks untuk diprediksi."
    analyzing_message = "Menganalisis sentimen ulasan Anda..."
    result_label = "🎉 Prediksi Sentimen: **{}**"
    confidence_label = "📊 Akurasi Prediksi: **{:.2f}**"
    usage_guide = """
    1. Masukkan ulasan Anda di kotak teks di atas.
    2. Klik tombol **Prediksi** untuk mengetahui hasil analisis sentimen.
    3. Nikmati pengalaman Anda di Universal Studios!
    """
else:
    # English
    st.markdown("<h1 style='text-align: center; color: black;'>🎥 Sentiment Analysis - Universal Studios Reviews</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        Welcome to the Universal Studios Review Sentiment Analysis app!<br> 
        Enter your review, and we will predict whether the sentiment is **Positive**, **Neutral**, or **Negative**.
        """,
        unsafe_allow_html=True,
    )
    user_input_placeholder = "Example: An amazing experience! All the rides were fun and exciting."
    prediction_button = "🎯 Predict"
    warning_message = "❌ Please enter text for prediction."
    analyzing_message = "Analyzing your review sentiment..."
    result_label = "🎉 Sentiment Prediction: **{}**"
    confidence_label = "📊 Accuracy Level: **{:.2f}**"
    usage_guide = """
    1. Enter your review in the text box above.
    2. Click the **Predict** button to see the sentiment analysis result.
    3. Enjoy your experience at Universal Studios!
    """

# Input teks dari pengguna
st.markdown("---")
user_input = st.text_area(
    "🌟 Masukkan Review Anda:" if language == "Bahasa Indonesia" else "🌟 Enter Your Review:",
    placeholder=user_input_placeholder,
    height=200,
)

if st.button(prediction_button):
    if user_input.strip() == "":
        st.warning(warning_message)
    else:
        with st.spinner(analyzing_message):
            sentiment, confidence = predict_sentiment(user_input, language=language)
        st.success(result_label.format(sentiment))
        st.info(confidence_label.format(confidence))

# Panduan penggunaan
with st.expander("ℹ️ Panduan Penggunaan" if language == "Bahasa Indonesia" else "ℹ️ Usage Guide"):
    st.write(usage_guide)

st.markdown("---")
st.markdown(
    "💡 **Catatan:** Aplikasi ini menggunakan model LSTM untuk menganalisis sentimen berdasarkan ulasan teks."
    if language == "Bahasa Indonesia"
    else "💡 **Note:** This application uses an LSTM model to analyze sentiment based on textual reviews."
)

# Menambahkan footer dengan informasi tambahan atau kontak
if language == "Bahasa Indonesia":
    st.markdown(
        """
        ---
        <div style="text-align: center;">
            &copy; 2024 oleh <strong>Mufidatul Nabilla Khasanah</strong> - <strong>202110370311227</strong><br>
            Disusun untuk memenuhi Tugas Ujian Akhir Praktikum Pembelajaran Mesin.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        ---
        <div style="text-align: center;">
            &copy; 2024 by <strong>Mufidatul Nabilla Khasanah</strong> - <strong>202110370311227</strong><br>
            Prepared to fulfill Machine Learning Practical Final Exam Assignment.
        </div>
        """,
        unsafe_allow_html=True,
    )
