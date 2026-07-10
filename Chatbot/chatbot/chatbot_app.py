# === FILE: chatbot_app.py ===
# Chatbot Jadwal Kuliah dengan Streamlit

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import json
import re

# Inisialisasi
stemmer = PorterStemmer()
nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Load model
data = torch.load("data.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load dataset jadwal kuliah untuk respon
jadwal_df = pd.read_csv("dataset_jadwal_kuliah.csv")

# Response generator berdasarkan intent dan entitas
import random

def get_response(intent, entity):
    if intent == "jadwal_hari_ini":
        hari_ini = pd.Timestamp.today().day_name()
        result = jadwal_df[jadwal_df['hari'].str.lower() == hari_ini.lower()]
    elif intent == "jadwal_hari_tertentu" and entity.get("hari"):
        result = jadwal_df[jadwal_df['hari'].str.lower() == entity["hari"].lower()]
    elif intent == "mata_kuliah_semester_tipe" and entity.get("semester_tipe"):
        result = jadwal_df[jadwal_df['semester_tipe'].str.lower() == entity["semester_tipe"].lower()]
    elif intent == "mata_kuliah_berdasarkan_lokasi" and entity.get("lokasi"):
        result = jadwal_df[jadwal_df['lokasi'].str.lower() == entity["lokasi"].lower()]
    elif intent == "mata_kuliah_per_semester" and entity.get("semester"):
        result = jadwal_df[jadwal_df['semester'] == int(entity["semester"])]
    elif intent == "mata_kuliah_dosen" and entity.get("dosen"):
        result = jadwal_df[jadwal_df['dosen'].str.lower().str.contains(entity["dosen"].lower())]
    else:
        return "Maaf, saya belum dapat menemukan informasi yang sesuai."

    if not result.empty:
        list_kuliah = result.apply(
            lambda row: f"{row['hari']}: {row['mata_kuliah']} (Jam {row['jam_mulai']}–{row['jam_selesai']}, Lokasi: {row['lokasi']})",
            axis=1
        ).tolist()
        return "Berikut jadwalnya bro, kuliah bener2 lu tu:\n- " + "\n- ".join(list_kuliah)
    else:
        return "Tidak ditemukan jadwal yang sesuai."

# Ekstrak entitas manual dari teks

def extract_entity(text):
    hari_list = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
    semester_tipe = ["ganjil", "genap"]
    entity = {}

    for h in hari_list:
        if h in text.lower():
            entity["hari"] = h
            break
    for s in semester_tipe:
        if s in text.lower():
            entity["semester_tipe"] = s
            break
    match_semester = re.search(r"semester\s+(\d)", text.lower())
    if match_semester:
        entity["semester"] = match_semester.group(1)

    lokasi = jadwal_df["lokasi"].dropna().unique()
    for l in lokasi:
        if l.lower() in text.lower():
            entity["lokasi"] = l
            break

    dosen_list = jadwal_df["dosen"].dropna().unique()
    for d in dosen_list:
        if d.lower().split()[0] in text.lower():
            entity["dosen"] = d
            break

    return entity

# Prediksi intent

def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        return tag
    else:
        return "unknown"

# Streamlit App
st.title("🎓 Chatbot Jadwal Kuliah Semester Ganjil Tahun Ajaran 2025/2026")
st.markdown("Tanyakan tentang jadwal kuliah berdasarkan hari, dosen, semester, atau ruangan.")

user_input = st.text_input("Ketik pertanyaan kamu:", "Apa jadwal kuliah hari Senin?")

if st.button("Tanya"):
    intent = predict_class(user_input)
    entity = extract_entity(user_input)
    response = get_response(intent, entity)
    st.write(response)