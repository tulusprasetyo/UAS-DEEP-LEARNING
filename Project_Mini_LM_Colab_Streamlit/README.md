# Project Mini Language Model Berbasis Transformer

Isi paket:

- `dataset/` berisi dataset praktikum dalam format DOCX dan PDF.
- `notebook/01_training_mini_llm_colab.ipynb` untuk training di Google Colab.
- `notebook/training_mini_llm_colab.py` versi script dari notebook.
- `streamlit_app/` untuk menjalankan model hasil training secara lokal menggunakan Streamlit.

Alur:

1. Buka notebook di Google Colab.
2. Upload dataset DOCX/PDF.
3. Jalankan training dan evaluasi.
4. Download `mini_lm_model.zip`.
5. Salin ZIP ke `streamlit_app/model_artifacts/`.
6. Jalankan `streamlit run app.py` dari folder `streamlit_app`.
