# Mini Language Model Streamlit

Aplikasi ini memuat `mini_lm_model.zip` hasil training dari Google Colab.

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Letakkan file `mini_lm_model.zip` pada folder:

```text
model_artifacts/mini_lm_model.zip
```

Atau upload file ZIP langsung melalui antarmuka Streamlit.
