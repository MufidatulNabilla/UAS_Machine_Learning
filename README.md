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
  - Ukuran batch: 64, Epoch: 30. Berdasarkan panjang urutan dan ukuran batch.
- **Evaluasi Kinerja**:
  - Classification Report 🔍, Confusion Matrix 📋, Grafik Visualisai 📊
  
#### B. Model BERT
- **Proses**:
  - Tokenisasi teks menggunakan `BertTokenizer`.
  - Encoding label menggunakan `LabelEncoder`.
  - Fine-tuning `TFBertForSequenceClassification` yang sudah dilatih sebelumnya.
- **Pelatihan**:
  - Ukuran batch: 64, Epoch: 10. Menggunakan `AdamWeightDecay`.
- **Evaluasi Kinerja**: 
  - Classification Report 🔍, Confusion Matrix 📋, Grafik Visualisai 📊

---

## 📝Hasil Evaluasi dan Visualisasi
- **1️⃣ Laporan Klasifikasi dan Confusion Matrix**:
  - **LSTM**:
     - **Classification Report** 🔍  
       ![Classification Report LSTM](https://github.com/user-attachments/assets/f248a2a5-6d0b-4e83-bc78-92ec38451c37)
   
     - **Confusion Matrix** 📋  
       ![Confusion Matrix LSTM](https://github.com/user-attachments/assets/aacbda73-1b2d-4a4d-97bc-72cf08e5c564)
       
   - **BERT**:
     - **Classification Report** 🔍  
       ![Classification Report BERT](https://github.com/user-attachments/assets/395a6da2-ebce-442c-a586-0440c29a6ce0)
   
     - **Confusion Matrix** 📋  
       ![Confusion Matrix BERT](https://github.com/user-attachments/assets/e58733f7-6643-4e58-a337-30542953c355)
  
- **2️⃣ Visualisasi Hasil Pelatihan**:
  - Grafik Visualisai 📊 **LSTM**:
    ![Grafik Visualisasi LSTM](https://github.com/user-attachments/assets/7116736c-4747-4e94-9864-421032c97006)
 
  - **Grafik Visualisasi** 📊 **BERT**:  
     - **Akurasi**  
       ![Grafik Visualisasi BERT Akurasi](https://github.com/user-attachments/assets/f5e47eed-4844-479e-858a-bb604f7b81b3)
   
     - **Loss**  
       ![Grafik Visualisasi BERT Loss](https://github.com/user-attachments/assets/c4351d78-e434-4598-963d-211da3bb42c4)
 
---

## 🛠️ Persiapan dan Instalasi
1. Unduh dataset dari Kaggle:
   ```bash
   kaggle datasets download -d dwiknrd/reviewuniversalstudio
   ```
2. Ekstrak dataset yang diunduh:
   ```bash
   unzip reviewuniversalstudio.zip
   ```
3. Instal pustaka yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hasil
   - **Model LSTM**: Model ini menunjukkan kemampuan yang baik dalam memahami data ulasan dengan **akurasi pengujian sebesar 75.8%**.  
     Namun, terdapat variasi performa untuk setiap kelas:
     - **Positif**: Precision 85%, Recall 90%, F1-score 87% (kinerja terbaik).  
     - **Negatif dan Netral**: Memiliki precision dan recall yang lebih rendah, menunjukkan bahwa model sedikit kesulitan dalam membedakan sentimen-sentimen ini.
   
   - **Model BERT**: Dengan pendekatan berbasis transformer, BERT mencapai **akurasi pengujian sebesar 81.0%**, yang lebih tinggi dibandingkan LSTM.  
     Performanya lebih baik terutama dalam kelas **Positif**:
     - **Positif**: Precision 87%, Recall 95%, F1-score 91% (unggul dalam memahami sentimen positif).  
     - **Negatif dan Netral**: Precision dan recall lebih rendah, tetapi tetap lebih baik dibandingkan LSTM.
   
   Secara keseluruhan, **BERT memberikan kinerja yang lebih konsisten dan akurat** dalam memahami konteks ulasan dibandingkan LSTM.

---

## 🌐 Deploy ke Website dengan Streamlit
Setelah model dievaluasi, proyek dapat diakses melalui aplikasi web interaktif yang dibangun menggunakan **Streamlit**. Aplikasi ini memungkinkan pengguna untuk menginput ulasan dan melihat prediksi sentimen secara langsung.

### Langkah-langkah untuk Menjalankan Aplikasi Streamlit 🙌
1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Menjalankan Aplikasi:
   ```bash
   streamlit run app.py
   ```
### 🌟Struktur Aplikasi Streamlit
   - app.py: Script utama untuk menjalankan aplikasi, yang memuat model dan menyediakan antarmuka pengguna.
   - lstm_model.h5: Model yang telah dilatih, disimpan dalam format h5 untuk digunakan dalam aplikasi.
   - tokenizer.joblib: Komponen dalam pemrosesan teks yang telah dilatih. disimpan dalam format joblib untuk digunakan dalam aplikasi.

### Tampilan Web Deploy 💻
   <p align="center">
     <img src="https://github.com/user-attachments/assets/1b664453-1a68-485b-8471-995681990d26" width="45%" />
     <img src="https://github.com/user-attachments/assets/c3d7af21-ea3f-4af7-b659-1ae59974baf6" width="45%" />
   </p>

---

## 👨‍💻 Author 
[Mufidatul Nabilla Khasanah](https://github.com/MufidatulNabilla)

