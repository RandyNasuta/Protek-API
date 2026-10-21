import pandas as pd
import re
import torch
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json

print("Memuat dataset")
dataset_path = 'assets/turnbackhoax_dataset.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"File dataset tidak ditemukan: {dataset_path}"
    )
    
    
df = pd.read_csv(dataset_path, sep=';', encoding='latin1', on_bad_lines='skip')

print("\nDistribusi label asli:")
print(df["Label"].value_counts(dropna=False).sort_index())

print("\nContoh data untuk setiap label:")
for label in sorted(df["Label"].dropna().unique()):
    print(f"\n=== Label asli: {label} ===")

    samples = df.loc[
        df["Label"] == label,
        ["Headline", "Body"]
    ].head(3)

    for _, row in samples.iterrows():
        print("Judul:", str(row["Headline"])[:250])
        print("Isi:", str(row["Body"])[:400])
        print("-" * 50)

df = df.dropna(axis=1, how='all')
df = df.dropna(subset=['Label', 'Headline', 'Body'])

df['Full_Text'] = df['Headline'].astype(str) + ". " + df['Body'].astype(str)

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

for i in range(min(3, len(df))):
    print(f"\n--- Sampel {i + 1} ---")
    print("Label asli:", df.iloc[i]["Label"])
    print("Judul:", str(df.iloc[i]["Headline"])[:200])

    teks_asli = str(df.iloc[i]["Headline"]) + ". " + str(df.iloc[i]["Body"])
    teks_bersih = clean_hoax_text(teks_asli)

    print("Teks sebelum dibersihkan:")
    print(teks_asli[:500])

    print("\nTeks setelah dibersihkan:")
    print(teks_bersih[:500])

    print("\nPanjang teks bersih:", len(teks_bersih), "karakter")
    
print("\n=== UJI CLEAN_HOAX_TEXT ===")

contoh = (
    "Isi artikel menjelaskan hasil pemeriksaan fakta. "
    "====== KATEGORI Hoaks. "
    "====== SUMBER https://contoh.com/artikel"
)

hasil = clean_hoax_text(contoh)

print("Sebelum:", contoh)
print("Sesudah:", hasil)

print("Membersihkan teks dari data leakage")
df['clean_text'] = df['Full_Text'].apply(clean_hoax_text)

df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"]).copy()
df = df[df["Label"].isin([0, 1])].copy()
df["Label"] = df["Label"].astype(int)

# Target model: 0 = FAKTA, 1 = HOAKS
df["target"] = df["Label"].map({1: 0, 0: 1})

# Hapus teks kosong
df = df[df["clean_text"].str.strip().ne("")].copy()

data = df[["clean_text", "target"]].copy()

print("Distribusi label target:")
print(data["target"].value_counts().sort_index())

data["_row_id"] = df.index

train_val_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["target"],
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,
    random_state=42,
    stratify=train_val_df["target"],
)

test_indices = test_df["_row_id"].tolist()

save_path = "model_saved/indobert-hoax-model"
os.makedirs(save_path, exist_ok=True)
    
train_df = train_df.drop(columns=["_row_id"])
val_df = val_df.drop(columns=["_row_id"])
test_df = test_df.drop(columns=["_row_id"])

train_df = train_df.rename(
    columns={"clean_text": "text", "target": "label"}
).reset_index(drop=True)

val_df = val_df.rename(
    columns={"clean_text": "text", "target": "label"}
).reset_index(drop=True)

test_df = test_df.rename(
    columns={"clean_text": "text", "target": "label"}
).reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

print(f"Data training: {len(train_df)}")
print(f"Data validasi: {len(val_df)}")
print(f"Data testing: {len(test_df)}")

print("Memuat Tokenizer & Model IndoBERT")
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

print("Setup Training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Perangkat komputasi: {device.type.upper()}")


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={
        0: "BUKAN_HOAKS",
        1: "HOAKS"
    },
    label2id={
        "BUKAN_HOAKS": 0,
        "HOAKS": 1
    },
    ignore_mismatched_sizes=True
).to(device)

training_args = TrainingArguments(
    output_dir="./hoax_model_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print("Mulai Fine-Tuning IndoBERT (Mohon tunggu)")
trainer.train()

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

with open(
    os.path.join(save_path, "test_indices.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(test_indices, f)

print("Model dan indeks testing berhasil disimpan.")
print("Proses training selesai!")