# model/infer.py

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== แก้ import ให้มองเห็น preprocessing.py แน่นอน ======
# เพิ่ม path ของโฟลเดอร์ project (โฟลเดอร์ที่มี main.py, preprocessing.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from preprocessing import clean_text  # ใช้ clean_text ตัวเดียวกับ preprocessing.py

LABELS = ["profanity", "sexual", "violence", "hate"]


# ===== 1) สร้าง context รอบ ๆ segment =====
def build_context(df: pd.DataFrame, window: int = 0) -> pd.DataFrame:
    """
    ถ้า window = 0 → ใช้เฉพาะ text เดี่ยว ๆ
    ถ้า window = 1 → ใช้ prev [SEP] curr [SEP] next เหมือนเวอร์ชันเดิม
    """
    texts = df["text"].astype(str).tolist()
    ctxs = []

    if window <= 0:
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


# ===== 2) โหลด threshold แบบยืดหยุ่น =====
def load_thresholds(model_dir: str) -> np.ndarray:
    """
    thresholds_per_label.json อาจเก็บได้ 2 แบบ:
    1) {"thresholds": [.. list ตามลำดับ LABELS ..]}
    2) {"profanity": 0.4, "sexual": 0.7, ...}  (dict ตามชื่อ label)

    ฟังก์ชันนี้จะคืน np.array(thresholds) ตามลำดับ LABELS เสมอ
    """
    thr_path = os.path.join(model_dir, "thresholds_per_label.json")
    print(f"📥 Loading thresholds from: {thr_path}")

    with open(thr_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    thr_dict = {}

    if isinstance(cfg, dict) and "thresholds" in cfg:
        # กรณีเก็บเป็น list ตรง ๆ
        arr = np.array(cfg["thresholds"], dtype=float)
        if arr.shape[0] != len(LABELS):
            raise ValueError(
                f"จำนวน thresholds ({arr.shape[0]}) ไม่ตรงกับ LABELS ({len(LABELS)})"
            )
        thr_dict = dict(zip(LABELS, arr))
    elif isinstance(cfg, dict):
        # กรณีเก็บเป็น dict ตาม label
        for lab in LABELS:
            if lab not in cfg:
                raise ValueError(
                    f"thresholds_per_label.json ไม่มี key สำหรับ label '{lab}'"
                )
            thr_dict[lab] = float(cfg[lab])
    else:
        raise ValueError("thresholds_per_label.json มีโครงสร้างไม่ถูกต้อง")

    thrs = np.array([thr_dict[lab] for lab in LABELS], dtype=float)
    print("🔧 Per-label thresholds:", thr_dict)
    return thrs


# ===== 3) main infer =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="เช่น tox_ft\\best_model หรือ tox_ft_pos\\best_model",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="ไฟล์ transcript CSV (เช่น start_time,end_time,text)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="ไฟล์ผลลัพธ์รวมคะแนน (CSV)",
    )
    ap.add_argument(
        "--flag_out",
        required=True,
        help="ไฟล์เฉพาะแถวที่โดน flag อย่างน้อย 1 label",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="max sequence length สำหรับ tokenizer",
    )
    ap.add_argument(
        "--context_window",
        type=int,
        default=0,
        help="0 = ไม่ใช้ context, 1 = ใช้ prev/next + [SEP]",
    )
    args = ap.parse_args()

    # ----- โหลดโมเดล -----
    print(f"📦 Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # ----- โหลด thresholds -----
    thrs = load_thresholds(args.model_dir)

    # ----- โหลด CSV -----
    print(f"📄 Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv).fillna("")
    lower_cols = {c.lower(): c for c in df.columns}

    if "text" in lower_cols:
        text_col = lower_cols["text"]
    elif "transcript" in lower_cols:
        text_col = lower_cols["transcript"]
    else:
        raise ValueError("CSV ต้องมีคอลัมน์ 'text' หรือ 'transcript'")

    # ----- ทำความสะอาด text ด้วย clean_text จาก preprocessing.py -----
    print("🧹 Cleaning text...")
    df = df.copy()
    df["text"] = df[text_col].astype(str).apply(clean_text)

    # ----- สร้าง context -----
    print(f"🧱 Building context (window={args.context_window})...")
    df = build_context(df, window=args.context_window)
    texts = df["text_ctx"].astype(str).tolist()

    # ----- Tokenization -----
    print("🤖 Running inference...")
    enc = tokenizer(
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

    # ----- Sigmoid → prob -----
    probs = 1.0 / (1.0 + np.exp(-logits))

    # broadcast thresholds ให้เท่ารูป logits (N, num_labels)
    preds = (probs >= thrs[None, :]).astype(int)

    # ----- เขียนผลกลับลง df -----
    for i, lab in enumerate(LABELS):
        df[lab] = preds[:, i]
        df[f"{lab}_prob"] = probs[:, i]

    # ----- เซฟไฟล์ -----
    out_dir = os.path.dirname(args.out)
    flag_dir = os.path.dirname(args.flag_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if flag_dir:
        os.makedirs(flag_dir, exist_ok=True)

    df.to_csv(args.out, index=False)
    flagged = df[df[LABELS].sum(axis=1) > 0]
    flagged.to_csv(args.flag_out, index=False)

    print("✅ Saved full results to:", args.out)
    print("✅ Saved flagged rows to:", args.flag_out)
    print("🔢 Flagged segments:", len(flagged))


if __name__ == "__main__":
    main()
