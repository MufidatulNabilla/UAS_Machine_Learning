# Proyek Analisis Sentimen 🌟

Selamat datang di proyek **Analisis Sentimen Ulasan Universal Studios**! 🎢🎡 Repositori ini menerapkan model deep learning canggih untuk mengklasifikasikan ulasan pengunjung ke dalam sentimen: Positif, Netral, atau Negatif.

---

## ✨ Sorotan Proyek
1. **Dataset**: Ulasan pengunjung dari Universal Studios 🏙️, tersedia di Kaggle [di sini](https://www.kaggle.com/datasets/dwiknrd/reviewuniversalstudio).
2. **Model yang Digunakan**:
   - **LSTM** (Long Short-Term Memory): Memahami pola berurutan dalam data teks atau angka.
   - **BERT** (Bidirectional Encoder Representations from Transformers): Memanfaatkan embedding kontekstual untuk klasifikasi teks yang mutakhir.
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
  - Lapisan LSTM Bidirectional.
  - Lapisan Dense yang Terhubung Penuh.
- **Pelatihan**:
  - Ukuran batch: 32, Epoch: 15.
  - Dioptimalkan untuk panjang urutan dan ukuran batch.
- **Kinerja**:
  - Metrik: Akurasi, Presisi, Recall.

#### B. Model BERT
- **Proses**:
  - Tokenisasi teks menggunakan `BertTokenizer`.
  - Encoding label menggunakan `LabelEncoder`.
  - Fine-tuning `TFBertForSequenceClassification` yang sudah dilatih sebelumnya.
- **Pelatihan**:
  - Ukuran batch: 64, Epoch: 10.
  - Dioptimalkan dengan `AdamWeightDecay`.
- **Evaluasi**:
  - Confusion Matrix 📊
  - Laporan Klasifikasi 📋

---

## 🔍 Visualisasi
- **Perkembangan Pelatihan**:
  - Tren Akurasi dan Loss per epoch untuk kedua model.
- **Confusion Matrix**:
  - Ditampilkan dengan heatmap untuk evaluasi yang jelas.

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

