import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# Config names and vars
# -----------------------------
TRAIN_FILE = "train_30k.csv"
TEST_FILE  = "test_3k.csv"

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 256
RANDOM_SEED = 42

# -----------------------------
# Read CSV
# -----------------------------

ALLOWED_LABELS = {"negative", "neutral", "positive"}

def _clean_outer_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s

def load_dataset(csv_path):
    """
    Reads a file in the format:

    CONTENT;SENTIMENT
    'Texto ...';neutral
    "Outro texto ...";positive
    ...

    Supports:
        - reviews spanning multiple lines
        - ';' within the text
        - single/double quotes within the content
    """
    texts = []
    sentiments = []

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()

        buffer_text = None
        buffer_label = None

        for lineno, raw_line in enumerate(f, start=2):
            line = raw_line.rstrip("\n")

            if not line.strip():
                continue

            cand_text = None
            cand_label = None

            if ";" in line:
                before, after = line.rsplit(";", 1)
                label_candidate = after.strip().strip("'\"").lower()

                if label_candidate in ALLOWED_LABELS:
                    cand_text = before
                    cand_label = label_candidate

            if cand_label is not None:
                if buffer_text is not None and buffer_label is not None:
                    texts.append(_clean_outer_quotes(buffer_text))
                    sentiments.append(buffer_label)

                buffer_text = cand_text
                buffer_label = cand_label
            else:
                if buffer_text is None:
                    buffer_text = line
                    buffer_label = None
                else:
                    buffer_text += "\n" + line

        if buffer_text is not None and buffer_label is not None:
            texts.append(_clean_outer_quotes(buffer_text))
            sentiments.append(buffer_label)

    df = pd.DataFrame({"text": texts, "sentiment": sentiments})
    return df

print("Lendo arquivos...")
train_full_df = load_dataset(TRAIN_FILE)
test_df       = load_dataset(TEST_FILE)

# -----------------------------
# Map values
# -----------------------------
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def map_labels(df):
    df = df.copy()
    df["label"] = df["sentiment"].map(label_map)
    if df["label"].isnull().any():
        missing = df[df["label"].isnull()]["sentiment"].unique()
        raise ValueError(f"Rótulos desconhecidos encontrados: {missing}")
    return df

train_full_df = map_labels(train_full_df)
test_df       = map_labels(test_df)

# -----------------------------
# Split train / valid
# -----------------------------
print("Splitting training and validation (80/20, stratified by class)...")

train_df, valid_df = train_test_split(
    train_full_df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=train_full_df["label"]
)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
print(f"Fixed test size: {len(test_df)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

def to_hf_dataset(df):
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")
    return ds

train_ds = to_hf_dataset(train_df)
valid_ds = to_hf_dataset(valid_df)
test_ds  = to_hf_dataset(test_df)

# -----------------------------
# Model and Trainer
# -----------------------------
print("Loading BERTimbau model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    f1 = f1_score(labels, preds, average="weighted")
    return {"weighted_f1": f1}

training_args = TrainingArguments(
    output_dir="./bertimbau-sentiment-pt",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()

print("Evaluation in validation:")
valid_metrics = trainer.evaluate(valid_ds)
print(valid_metrics)

print("\n===== TEST RESULTS (3k) =====")
test_metrics = trainer.evaluate(test_ds)
for k, v in test_metrics.items():
    try:
        print(f"{k}: {v:.4f}")
    except TypeError:
        print(f"{k}: {v}")
