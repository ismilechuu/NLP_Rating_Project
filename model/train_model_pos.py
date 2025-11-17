# model/train_model_pos.py

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ---- 1) config เบื้องต้น ----
label_cols = ["profanity", "sexual", "violence", "hate"]

def clean_text(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("f*ck", "fuck").replace("sh*t", "shit").replace("a**", "ass")
    s = re.sub(r"\s+", " ", s)
    return s

# ---- 2) โหลดและเตรียม dataset ----
DATA_PATH = r"C:\Users\SMILEGURL\Documents\work\cs374\Rating\model\Data\train.csv"

df = pd.read_csv(DATA_PATH)

# mapping labels จาก Jigsaw
df["profanity"] = ((df["obscene"] == 1) | (df["insult"] == 1)).astype(int)
df["sexual"]    = (df["obscene"] == 1).astype(int)
df["violence"]  = (df["threat"] == 1).astype(int)
df["hate"]      = (df["identity_hate"] == 1).astype(int)

df["text"] = df["comment_text"].apply(clean_text)
df = df[["text"] + label_cols]

# split train/valid
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)

print("Train size:", len(train_df), "Valid size:", len(valid_df))

# ---- 3) คำนวณ pos_weight (สำคัญ) ----
N = len(train_df)
pos_weight_list = []
for lab in label_cols:
    P = int(train_df[lab].sum())
    w = (N - P) / max(P, 1)   # กันหารศูนย์
    pos_weight_list.append(w)
pos_weight = torch.tensor(pos_weight_list, dtype=torch.float)
print("pos_weight:", dict(zip(label_cols, [float(x) for x in pos_weight])))

# ---- 4) สร้าง tokenizer + dataset tokenized ----
tok = AutoTokenizer.from_pretrained("distilroberta-base")

def encode(batch, max_len=256):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = [
        [float(batch[c][i]) for c in label_cols]
        for i in range(len(batch["text"]))
    ]
    return enc

train_tok = Dataset.from_pandas(train_df).map(encode, batched=True)
valid_tok = Dataset.from_pandas(valid_df).map(encode, batched=True)

# ---- 5) โหลดโมเดล ----
model = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=len(label_cols),
    problem_type="multi_label_classification",
)

# ---- 6) สร้าง WeightedTrainer ----
class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # ตัด labels ออกตอนส่งเข้าโมเดล
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# ---- 7) TrainingArguments ----
args = TrainingArguments(
    output_dir="tox_ft_pos",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,                  # ลอง 2 epoch พอ (ไวกว่า 3)
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    load_best_model_at_end=True,
    #metric_for_best_model="f1_macro",
    metric_for_best_model="loss", # ⭐ ใช้ loss แทน 
    greater_is_better=False,
    logging_steps=100,
)


# ---- 8) สร้าง trainer และเทรน ----
trainer = WeightedTrainer(
    pos_weight=pos_weight,
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    tokenizer=tok,
)

trainer.train()

# ---- 9) เซฟโมเดลใหม่ ----
save_dir = "tox_ft_pos/best_model"
trainer.save_model(save_dir)
tok.save_pretrained(save_dir)
print("✅ Training with pos_weight finished & model saved to", save_dir)
