# 1) import และตั้งค่าเบื้องต้น
import re, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

model_name = "distilroberta-base"
label_cols = ["profanity","sexual","violence","hate"]

def clean_text(s:str):

    s = str(s).strip().lower()
    # decensor ที่พบบ่อย
    patterns = {
    r"f[\*\-_\.\s]?u[\*\-_\.\s]?c[\*\-_\.\s]?k": "fuck",
    r"s[\*\-_\.\s]?h[\*\-_\.\s]?i[\*\-_\.\s]?t": "shit",
    r"b[\*\-_\.\s]?i[\*\-_\.\s]?t[\*\-_\.\s]?c[\*\-_\.\s]?h": "bitch",
    r"d[\*\-_\.\s]?i[\*\-_\.\s]?c[\*\-_\.\s]?k": "dick",
    r"p[\*\-_\.\s]?u[\*\-_\.\s]?s[\*\-_\.\s]?s[\*\-_\.\s]?y": "pussy",
    r"c[\*\-_\.\s]?u[\*\-_\.\s]?n[\*\-_\.\s]?t": "cunt"
    }
    for pat, rep in patterns.items():
        s = re.sub(pat, rep, s)
    return s

# 2) โหลด Jigsaw แล้วทำ mapping 4 ป้าย
df = pd.read_csv(r"C:\Users\SMILEGURL\Documents\work\cs374\Rating\model\Data\train.csv") 

df["profanity"] = ((df["obscene"]==1) | (df["insult"]==1)).astype(int)
df["sexual"]    = (df["obscene"]==1).astype(int)
df["violence"]  = (df["threat"]==1).astype(int)
df["hate"]      = (df["identity_hate"]==1).astype(int)

df = df[["comment_text"] + label_cols].dropna()
df["text"] = df["comment_text"].apply(clean_text)
df = df.drop(columns=["comment_text"])

# 3) split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# 4) tokenizer + dataset
tok = AutoTokenizer.from_pretrained(model_name)

def to_ds(pdf):
    return Dataset.from_pandas(pdf.reset_index(drop=True))

train_ds, valid_ds, test_ds = map(to_ds, [train_df, valid_df, test_df])

def encode(batch, max_len=256):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
    # เดิม: int → ทำให้กลายเป็น LongTensor
    # enc["labels"] = [[int(batch[c][i]) for c in label_cols] for i in range(len(batch["text"]))]

    # ใหม่: float32 → ใช้ได้กับ BCEWithLogitsLoss
    enc["labels"] = [
        [float(batch[c][i]) for c in label_cols]
        for i in range(len(batch["text"]))
    ]
    return enc


train_tok = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
valid_tok = valid_ds.map(encode, batched=True, remove_columns=valid_ds.column_names)
test_tok  = test_ds .map(encode, batched=True, remove_columns=test_ds .column_names)

# 5) โหลดโมเดล + ตั้งค่าฝึก
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"  # sigmoid + BCEWithLogitsLoss
)

'''def compute_metrics(eval_pred, thr=0.5):
    logits, labels = eval_pred
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= thr).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }'''

args = TrainingArguments(
    output_dir="tox_ft",
    eval_strategy="epoch",          # ← เปลี่ยนชื่อพารามิเตอร์
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    # ถ้าจะใช้ enum ก็ได้:
    # eval_strategy=IntervalStrategy.EPOCH,
    # save_strategy=IntervalStrategy.EPOCH,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=valid_tok,
    tokenizer=tok
)

trainer.train()


'''# 6) จูน threshold ต่อป้าย (ง่ายๆ เท่ากันทุกป้ายก่อน)
logits_valid = trainer.predict(valid_tok).predictions
labels_valid = np.array([valid_df[c].values for c in label_cols]).T
probs_valid  = 1/(1+np.exp(-logits_valid))

candidates = np.linspace(0.3, 0.7, 9)
best_thr, best_macro = 0.5, -1
for t in candidates:
    preds = (probs_valid >= t).astype(int)
    macro = f1_score(labels_valid, preds, average="macro", zero_division=0)
    if macro > best_macro:
        best_macro, best_thr = macro, float(t)

# (ออปชัน) จูนทีละป้ายเพิ่มความเนียนในภายหลัง'''

# ประเมินบน test
'''logits_test = trainer.predict(test_tok).predictions
labels_test = np.array([test_df[c].values for c in label_cols]).T
probs_test  = 1/(1+np.exp(-logits_test))
preds_test  = (probs_test >= best_thr).astype(int)

print("Best shared threshold:", best_thr)
print(classification_report(labels_test, preds_test, target_names=label_cols, zero_division=0))
'''
# เซฟโมเดล + threshold
trainer.save_model("tox_ft/best_model")
tok.save_pretrained("tox_ft/best_model")
'''with open("tox_ft/best_model/thresholds.json","w") as f:
    json.dump({"labels": label_cols, "threshold": best_thr}, f, indent=2)'''
