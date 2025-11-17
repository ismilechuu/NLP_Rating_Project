# tune_and_eval.py
import json, numpy as np, pandas as pd
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset

label_cols = ["profanity","sexual","violence","hate"]

# โหลดโมเดล
#model_dir = "tox_ft/best_model"
model_dir = "tox_ft_pos/best_model"

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# โหลด valid/test (ต้องใช้ไฟล์เดียวกับตอนเทรน)
df = pd.read_csv(r"C:\Users\SMILEGURL\Documents\work\cs374\Rating\model\Data\train.csv")

df["profanity"] = ((df["obscene"]==1) | (df["insult"]==1)).astype(int)
df["sexual"]    = (df["obscene"]==1).astype(int)
df["violence"]  = (df["threat"]==1).astype(int)
df["hate"]      = (df["identity_hate"]==1).astype(int)

df["text"] = df["comment_text"].astype(str).str.lower()

# split
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

def encode(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=256)
    enc["labels"] = [[float(batch[c][i]) for c in label_cols] for i in range(len(batch["text"]))]
    return enc

valid_tok = Dataset.from_pandas(valid_df).map(encode, batched=True)
test_tok  = Dataset.from_pandas(test_df).map(encode, batched=True)

trainer = Trainer(model=model)

# จูน threshold
logits_valid = trainer.predict(valid_tok).predictions
labels_valid = valid_df[label_cols].values
probs_valid  = 1/(1+np.exp(-logits_valid))

best_thr = []
for i, lab in enumerate(label_cols):
    best_f1, best_t = -1, 0.5
    for t in np.linspace(0.2,0.8,25):
        f1 = f1_score(labels_valid[:,i], (probs_valid[:,i]>=t), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thr.append(best_t)

print("Best thresholds:", dict(zip(label_cols, best_thr)))

# evaluate on test
logits_test = trainer.predict(test_tok).predictions
probs_test  = 1/(1+np.exp(-logits_test))
preds_test  = np.array([(probs_test[:,i]>=best_thr[i]).astype(int) for i in range(len(label_cols))]).T

print("\n===== TEST REPORT =====")
print(classification_report(test_df[label_cols], preds_test, target_names=label_cols, zero_division=0))

with open(f"{model_dir}/thresholds_per_label.json","w") as f:
    json.dump({"labels": label_cols, "thresholds": best_thr}, f, indent=2)

print("✅ Thresholds saved!")
