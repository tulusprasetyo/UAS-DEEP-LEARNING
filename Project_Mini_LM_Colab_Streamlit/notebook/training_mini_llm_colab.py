# Praktikum Language Model Mini Berbasis Transformer
# Training di Google Colab

!pip install -q python-docx pypdf

from google.colab import files
uploaded = files.upload()
DATASET_PATH = list(uploaded.keys())[0]
print("Dataset:", DATASET_PATH)

import re, json, math, zipfile
from docx import Document
from pypdf import PdfReader
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

if DATASET_PATH.lower().endswith(".docx"):
    raw_text = read_docx(DATASET_PATH)
elif DATASET_PATH.lower().endswith(".pdf"):
    raw_text = read_pdf(DATASET_PATH)
else:
    raise ValueError("Gunakan dataset .docx atau .pdf")

text = clean_text(raw_text)
text = (text + "\n") * 60
print("Jumlah karakter:", len(text))
print(text[:700])

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

print("Vocabulary size:", vocab_size)

# Hyperparameter
block_size = 96
batch_size = 64
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.2
learning_rate = 1e-3
max_iters = 1200
eval_interval = 100
eval_iters = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_source = train_data if split == "train" else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        _, T, _ = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B*T, -1), targets.reshape(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=20):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

model = MiniGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
print(sum(p.numel() for p in model.parameters()), "parameters")

@torch.no_grad()
def estimate_loss(eval_iters=50):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(eval_iters)
        print(f"step {step:4d}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss(eval_iters=100)
perplexity = math.exp(losses["val"])
print("Train loss:", losses["train"])
print("Validation loss:", losses["val"])
print("Perplexity:", perplexity)

for prompt in ["deep learning adalah", "transformer adalah", "cnn digunakan untuk", "rnn digunakan untuk"]:
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    output = model.generate(context, max_new_tokens=350, temperature=0.8, top_k=20)
    print("\nPROMPT:", prompt)
    print(decode(output[0].tolist()))

metadata = {
    "stoi": stoi,
    "itos": {str(k): v for k, v in itos.items()},
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "train_loss": losses["train"],
    "val_loss": losses["val"],
    "perplexity": perplexity,
}

torch.save(model.state_dict(), "mini_lm_weights.pth")
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

with zipfile.ZipFile("mini_lm_model.zip", "w") as z:
    z.write("mini_lm_weights.pth")
    z.write("metadata.json")

files.download("mini_lm_model.zip")
