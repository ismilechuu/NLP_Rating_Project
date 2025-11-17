# model/infer_v2.py

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["profanity", "sexual", "violence", "hate"]

# ===== 1) Advanced clean_text (เวอร์ชันนุ่มนวลกว่าเดิม) =====

LEET_MAP = str.maketrans({
    "@": "a",
    "$": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "3": "e",
    "4": "a",
    "7": "t",
})

PROFANITY_PATTERNS = [
    (r"f[\W_]*u[\W_]*c[\W_]*k+", "fuck"),
    (r"f+[\W_]*\*+[\W_]*k+", "fuck"),
    (r"s[\W_]*h[\W_]*i[\W_]*t+", "shit"),
    (r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+", "bitch"),
    (r"a[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e+", "asshole"),
    (r"a[\W_]*s[\W_]*s+", "ass"),
    (r"d[\W_]*a[\W_]*m[\W_]*n+", "damn"),
    (r"p[\W_]*o[\W_]*r[\W_]*n+", "porn"),
]

URL_RE = re.compile(r"http\S+|www\.\S+")

def clean_text_advanced(s: str) -> str:
    s = str(s).strip()
    # normalize quotes
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.lower()
    # remove URL
    s = URL_RE.sub(" ", s)
    # leet
    s = s.translate(LEET_MAP)
    # remove emoji / high unicode (แต่ไม่ยุ่งกับ ! . , ?)
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)
    # ลบ censor กลางคำแบบนุ่มนวล เช่น f***, b.i.t.c.h แต่ไม่ทำให้คำติดกันมั่ว
    s = re.sub(r"(?<=\w)[\*_]+(?=\w)", "", s)   # ลบ * / _ ระหว่างตัวอักษร
    s = re.sub(r"(?<=\w)\.(?=\w)", "", s)       # ลบ . ระหว่างตัวอักษร (b.i.t.c.h -> bitch)

    # apply profanity patterns
    for pat, repl in PROFANITY_PATTERNS:
        s = re.sub(pat, repl, s)

    # normalize space รอบ punctuation หลัก
    s = re.sub(r"\s+([,\.!?])", r"\1", s)      # space ก่อน , . ! ?
    s = re.sub(r"([,\.!?])([^\s])", r"\1 \2", s)  # space หลัง , . ! ?
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===== 2) context window (ปรับได้จาก args) =====

def build_context(df: pd.DataFrame, window: int = 0) -> pd.DataFrame:
    """
    ถ้า window=0 → ใช้เฉพาะ text เดี่ยว ๆ
    ถ้า window=1 → ใช้ prev [SEP] curr [SEP] next (เหมือนเวอร์ชันเดิม)
    """
    texts = df["text"].astype(str).tolist()
    ctxs = []
    if window <= 0:
        # no context
        ctxs = texts
    else:
        for i, t in enumerate(texts):
            prev_t = texts[i - 1] if i - 1 >= 0 else ""
            next_t = texts[i + 1] if i + 1 < len(texts) else ""
            ctx = prev_t + " [SEP] " + t + " [SEP] " + next_t
            ctxs.append(ctx)
    out = df.copy()
    out["text_ctx"] = ctxs
    return out

# ===== 3) main =====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="เช่น tox_ft\\best_model หรือ tox_ft_pos\\best_model")
    ap.add_argument("--csv", required=True,
                    help="ไฟล์ transcript CSV (start_time,end_time,text)")
    ap.add_argument("--out", required=True,
                    help="ไฟล์ผลลัพธ์รวมคะแนน")
    ap.add_argument("--flag_out", required=True,
                    help="ไฟล์เฉพาะแถวที่โดน flag")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--context_window", type=int, default=0,
                    help="0 = ไม่ใช้ context, 1 = ใช้ prev/next + [SEP]")
    args = ap.parse_args()

    print(f"📦 Loading model from: {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    thr_path = f"{args.model_dir}/thresholds_per_label.json"
    print(f"📥 Loading thresholds from: {thr_path}")
    with open(thr_path, "r") as f:
        cfg = json.load(f)
    thrs = np.array(cfg["thresholds"], dtype=float)
    print("🔧 Per-label thresholds:", dict(zip(LABELS, thrs)))

    print(f"📄 Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv).fillna("")
    lower_cols = {c.lower(): c for c in df.columns}

    if "text" in lower_cols:
        text_col = lower_cols["text"]
    elif "transcript" in lower_cols:
        text_col = lower_cols["transcript"]
    else:
        raise ValueError("CSV ต้องมีคอลัมน์ 'text' หรือ 'transcript'")

    print("🧹 Cleaning text...")
    df = df.copy()
    df["text"] = df[text_col].astype(str).apply(clean_text_advanced)

    # context
    print(f"🧱 Building context (window={args.context_window})...")
    df = build_context(df, window=args.context_window)
    texts = df["text_ctx"].astype(str).tolist()

    # tokenization
    print("🤖 Running inference...")
    enc = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=args.max_len,
        return_tensors="pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thrs).astype(int)

    for i, lab in enumerate(LABELS):
        df[lab] = preds[:, i]
        df[f"{lab}_prob"] = probs[:, i]

    df.to_csv(args.out, index=False)
    flagged = df[df[LABELS].sum(axis=1) > 0]
    flagged.to_csv(args.flag_out, index=False)

    print("✅ Saved full results to:", args.out)
    print("✅ Saved flagged rows to:", args.flag_out)
    print("🔢 Flagged segments:", len(flagged))


if __name__ == "__main__":
    main()

# python model\infer.py --model_dir tox_ft\best_model --csv "Transcript\STOP BEING A PEOPLE PLEASER - Gary Vaynerchuk Motivation.csv" --out outputs/transcript_with_scores.csv --flag_out outputs/transcript_flagged.csv --context_window 0