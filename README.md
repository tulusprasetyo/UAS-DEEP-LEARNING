# 🎓 UAS Deep Learning - Kumpulan Project

Repositori ini berisi kumpulan tugas dan proyek akhir (UAS) untuk mata kuliah Deep Learning. Proyek-proyek di bawah ini diimplementasikan menggunakan arsitektur deep learning serta deployment berbasis web.

---

## 📂 Struktur Repositori

Repositori ini menggunakan struktur monorepo yang memisahkan setiap project ke dalam foldernya masing-masing:

*   **`Chatbot/`** : Implementasi sistem chatbot pintar menggunakan arsitektur Deep Learning (Neural Networks).
*   **`Project_Mini_LM_Colab_Streamlit/`** : Implementasi Mini Language Model berbasis arsitektur **Transformer** yang dilatih di Google Colab dan di-deploy secara lokal menggunakan **Streamlit**.

---

## 🚀 Proyek 1: Chatbot Deep Learning
Aplikasi chatbot interaktif berbasis teks yang mampu memahami intent pengguna menggunakan klasifikasi teks dengan jaringan saraf tiruan.

### Fitur Utama:
*   Klasifikasi Intent berbasis Deep Learning.
*   Pemrosesan teks masukan (Preprocessing & Tokenization).
*   Antarmuka chatbot yang responsif.

---

## 🤖 Proyek 2: Mini Language Model Berbasis Transformer
Aplikasi web lokal yang dirancang untuk mencoba Mini Language Model yang sudah dilatih sebelumnya pada lingkungan Google Colab.

### Struktur Fitur:
*   **Dataset**: Dokumentasi dan teks latih yang digunakan pada proses pelatihan model.
*   **Notebook**: Dokumen `.ipynb` berisi proses training arsitektur Transformer di Google Colab.
*   **Streamlit Web App**: Aplikasi lokal untuk melakukan inferensi teks secara langsung lewat browser.

### Cara Menjalankan Aplikasi Web (Proyek 2):
1. Masuk ke direktori aplikasi web:
   ```bash
   cd Project_Mini_LM_Colab_Streamlit/streamlit_app


Pastikan file bobot model mini_lm_model.zip hasil training Google Colab sudah diletakkan di dalam folder model_artifacts atau siap diunggah melalui antarmuka web.

    Jalankan aplikasi Streamlit:
    Bash

    streamlit run app.py

🛠️ Tech Stack & Library

    Language: Python

    Framework Inference/Web: Streamlit

    Deep Learning Framework: TensorFlow / PyTorch (Sesuai kebutuhan arsitektur model)

    Notebook Environment: Google Colab & Jupyter Notebook

💡 Dibuat untuk memenuhi tugas penilaian Final Semester (UAS) Mata Kuliah Deep Learning - Semester 6.