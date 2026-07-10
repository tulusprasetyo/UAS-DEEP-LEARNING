import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import torch

from model import MiniGPT

st.set_page_config(page_title="Mini Language Model", page_icon="🤖", layout="centered")

ARTIFACT_DIR = Path("model_artifacts")
DEFAULT_ZIP = ARTIFACT_DIR / "mini_lm_model.zip"

@st.cache_resource
def load_model_from_zip(zip_path: str, cache_buster: float):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        with open(tmpdir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MiniGPT(
            vocab_size=metadata["vocab_size"],
            block_size=metadata["block_size"],
            n_embd=metadata["n_embd"],
            n_head=metadata["n_head"],
            n_layer=metadata["n_layer"],
            dropout=metadata.get("dropout", 0.2),
        ).to(device)
        state = torch.load(tmpdir / "mini_lm_weights.pth", map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, metadata, device

def make_codec(metadata):
    stoi = metadata["stoi"]
    itos = {int(k): v for k, v in metadata["itos"].items()}
    fallback = " " if " " in stoi else next(iter(stoi.keys()))

    def encode(text: str):
        return [stoi.get(ch, stoi[fallback]) for ch in text]

    def decode(ids):
        return "".join([itos[int(i)] for i in ids])

    return encode, decode

st.title("Mini Language Model Berbasis Transformer")
st.write("Aplikasi lokal untuk mencoba model yang sudah dilatih di Google Colab.")

uploaded_model = st.file_uploader("Upload mini_lm_model.zip hasil training Colab", type=["zip"])

if uploaded_model is not None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACT_DIR / "uploaded_mini_lm_model.zip"
    model_path.write_bytes(uploaded_model.getvalue())
else:
    model_path = DEFAULT_ZIP

try:
    model, metadata, device = load_model_from_zip(str(model_path), model_path.stat().st_mtime)
    encode, decode = make_codec(metadata)
    st.success(f"Model berhasil dimuat. Device: {device}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Validation loss", f"{metadata.get('val_loss', 0):.4f}")
    col2.metric("Perplexity", f"{metadata.get('perplexity', 0):.2f}")
    col3.metric("Vocabulary", metadata.get("vocab_size", 0))

    prompt = st.text_input("Prompt", value="transformer adalah")
    max_new_tokens = st.slider("Panjang output", min_value=50, max_value=800, value=300, step=50)
    temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1)
    top_k = st.slider("Top-k", min_value=0, max_value=50, value=20, step=5)

    if st.button("Generate teks"):
        if not prompt.strip():
            st.warning("Prompt tidak boleh kosong.")
        else:
            idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
            with torch.no_grad():
                out = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                )
            generated = decode(out[0].tolist())
            st.subheader("Hasil")
            st.text_area("Output model", generated, height=300)

except FileNotFoundError:
    st.warning("Model belum ditemukan. Letakkan mini_lm_model.zip di folder model_artifacts atau upload melalui tombol di atas.")
    st.code("model_artifacts/mini_lm_model.zip")
except Exception as e:
    st.error("Terjadi error saat memuat model atau menjalankan aplikasi.")
    st.exception(e)
