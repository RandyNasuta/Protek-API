import os
import re
import numpy as np
import pandas as pd
import torch
import json

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

dataset_path = "assets/turnbackhoax_dataset.csv"
model_path = "model_saved/indobert-hoax-model"

print("Memuat dataset")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

df = pd.read_csv(
    dataset_path,
    sep=";",
    encoding="latin1",
    on_bad_lines="skip",
)

df = df.dropna(axis=1, how="all")
df = df.dropna(subset=["Label", "Headline", "Body"]).copy()

df["Full_Text"] = (
    df["Headline"].astype(str)
    + ". "
    + df["Body"].astype(str)
)


def clean_hoax_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " urltoken ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"={3,}\s*\(?\s*KATEGORI\s*\)?\s*:?\s*[^.=]*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"={3,}\s*\(?\s*SUMBER\s*\)?\s*:?.*?(?=\.|={3,}|$)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"={3,}\s*\(?\s*REFERENSI\s*\)?\s*:?.*?(?=\.|={3,}|$)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\[\s*(SALAH|HOAKS|DISINFORMASI|PENIPUAN|PALSU|BENAR|FAKTA)\s*\]",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"[^a-zA-Z0-9À-ÿ\s.,?!:;()'\"/-]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*[.]{2,}\s*", ". ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = text.strip(" .")

    return text

df["clean_text"] = df["Full_Text"].apply(clean_hoax_text)

df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"]).copy()
df = df[df["Label"].isin([0, 1])].copy()
df["Label"] = df["Label"].astype(int)

df["target"] = df["Label"].map({1: 0, 0: 1})

df = df[df["clean_text"].str.strip().ne("")].copy()

print("\nJumlah data per kelas:")
print(
    df["target"].map({
        0: "FAKTA",
        1: "HOAKS",
    }).value_counts()
)

index_path = os.path.join(model_path, "test_indices.json")

if not os.path.isfile(index_path):
    raise FileNotFoundError(
        f"File indeks testing tidak ditemukan: {index_path}. "
        "Latih ulang model menggunakan train_fakenews.py terlebih dahulu."
    )

with open(index_path, "r", encoding="utf-8") as f:
    test_indices = json.load(f)

missing_indices = set(test_indices) - set(df.index)

if missing_indices:
    raise ValueError(
        "Indeks testing tidak cocok dengan dataset. "
        "Pastikan dataset dan preprocessing sama seperti saat training."
    )

if not test_indices:
    raise ValueError("Daftar indeks testing kosong.")

if len(test_indices) != len(set(test_indices)):
    raise ValueError(
        "Terdapat indeks duplikat dalam test_indices.json."
    )

test_df = (
    df.loc[test_indices, ["clean_text", "target"]]
    .rename(columns={
        "clean_text": "text",
        "target": "label",
    })
    .reset_index(drop=True)
)

if len(test_df) == 0:
    raise ValueError("Dataset testing kosong.")

if set(test_df["label"].unique()) != {0, 1}:
    raise ValueError(
        "Data testing harus memuat kedua kelas: FAKTA dan HOAKS."
    )

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False,
)

print(f"\nJumlah data uji: {len(test_df)}")

print("\nMemuat model final")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_test = test_dataset.map(tokenize_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

args = TrainingArguments(
    output_dir="./evaluation_results",
    per_device_eval_batch_size=8,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
)

print("\nMulai evaluasi")
predictions = trainer.predict(tokenized_test)

y_true = np.array(test_df["label"])
y_pred = np.argmax(predictions.predictions, axis=-1)

print("\nHASIL EVALUASI")

print(f"Jumlah data uji: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["FAKTA", "HOAKS"],
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print("Urutan label: FAKTA (0), HOAKS (1)")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nEvaluasi selesai.")