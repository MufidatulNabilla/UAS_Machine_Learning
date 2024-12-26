# Analisis Sentimen Ulasan Pengunjung Universal Studio 

---

## Project Analisis Sentimen🌟

Selamat datang di proyek **Analisis Sentimen Ulasan Universal Studios**! 🎢🎡 Repositori ini menerapkan model deep learning untuk mengklasifikasikan ulasan pengunjung ke dalam sentimen: Positif, Netral, atau Negatif.

Dengan memahami sentimen yang terkandung dalam ulasan, proyek ini diharapkan dapat memberikan wawasan yang lebih baik bagi pengalaman pengunjung serta membantu dalam meningkatkan pelayanan.

---

## ✨ Overview Project
1. **Dataset**: Ulasan pengunjung dari Universal Studios 🏙️, tersedia di Kaggle [di sini](https://www.kaggle.com/datasets/dwiknrd/reviewuniversalstudio).
2. **Model yang Digunakan**:
   - **LSTM** (Long Short-Term Memory): Memahami pola berurutan dalam data teks atau angka.
   - **BERT** (Bidirectional Encoder Representations from Transformers): Memanfaatkan embedding kontekstual untuk klasifikasi teks yang akurat dan canggih.
3. **Tujuan**:
   - Melakukan analisis sentimen.
   - Membandingkan kinerja model.

---

## 📚 Alur Kerja

### 1️⃣ Persiapan Dataset
- **Sumber**: Dataset Ulasan Universal Studios.
- **Langkah Pra-pemrosesan**:
  - Menghapus URL, mention, hashtag, dan emoji.
  - Tokenisasi dan pembersihan teks.
  - Menerapkan stemming dan lemmatization.

### 2️⃣ Analisis Data Eksploratif (EDA)
- Visualisasi:
  - Distribusi penilaian 📊
  - Distribusi sentimen 🟢🟡🔴
  - Word cloud ulasan 🌥️
  - Tren di berbagai cabang dan tanggal 📈

### 3️⃣ Pendekatan Pemodelan

#### A. Model LSTM
- **Arsitektur**:
  - Tokenisasi dan padding.
  - Lapisan Embedding.
  - Lapisan LSTM.
  - Lapisan Dense.
- **Pelatihan**:
  - Ukuran batch: 32, Epoch: 15. Berdasarkan panjang urutan dan ukuran batch.
- **Kinerja**:
  - Classification Report 🔍
  - Confusion Matrix 📋
  - Grafik Visualisai 📊
  
#### B. Model BERT
- **Proses**:
  - Tokenisasi teks menggunakan `BertTokenizer`.
  - Encoding label menggunakan `LabelEncoder`.
  - Fine-tuning `TFBertForSequenceClassification` yang sudah dilatih sebelumnya.
- **Pelatihan**:
  - Ukuran batch: 64, Epoch: 10. Menggunakan `AdamWeightDecay`.
- **Evaluasi**: 
  - Classification Report 🔍
  - Confusion Matrix 📋
  - Grafik Visualisai 📊

---

## 📝Hasil Evaluasi dan Visualisasi
- **1️⃣ Laporan Klasifikasi dan Confusion Matrix**:
  - **LSTM**:
     - Classification Report 🔍
     - Confusion Matrix 📋
  - **BERT**:
     - Classification Report 🔍
     - Confusion Matrix 📋 
- **2️⃣ Visualisasi Hasil Pelatihan**:
  - Grafik Visualisai 📊 **LSTM**:     
  - Grafik Visualisai 📊 **BERT**:
     
---

## 🛠️ Persiapan dan Instalasi
1. Clone repositori:
   ```bash
   git clone <repository-url>
   ```
2. Instal pustaka yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```
3. Muat dataset dari Kaggle:
   ```bash
   kaggle datasets download -d dwiknrd/reviewuniversalstudio
   ```

---

## 🚀 Hasil
- **Model LSTM**: Menunjukkan kinerja yang tangguh dengan data sekuensial.
- **Model BERT**: Mencapai akurasi dan presisi yang unggul melalui pemahaman kontekstual.

---

## 🙌 Kontribusi
Silakan ajukan isu atau berkontribusi dengan mengirimkan pull request! Mari kita tingkatkan analisis sentimen bersama. 🌟

---

## 📧 Kontak
Untuk pertanyaan, hubungi [your-email@example.com].

