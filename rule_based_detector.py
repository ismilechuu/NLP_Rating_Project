# rule_based_detector.py
"""
Rule-based-only pipeline (ไม่ใช้โมเดล ML)

Flow:
    input.txt
      -> preprocessing.py  -> transcript_clean.csv
      -> rule-based tagging (lexicons.py) -> rule_full.csv + rule_flagged.csv
      -> rating_utils.rate_video(rule_full.csv) -> FINAL RATING
      -> rule_flagged.txt (rating + flagged segments)

Usage:
    # Full pipeline (with preprocessing and rating)
    python rule_based_detector.py --input test.txt

    # Inference only (for evaluation - skip preprocessing and rating)
    python rule_based_detector.py --csv input.csv --out output.csv
"""

import argparse
import os
import re
from pathlib import Path
import subprocess

import pandas as pd

from detoxify_detector import seconds_to_hms, _to_seconds
import rating_utils
import lexicons   # ใช้คำจากไฟล์ lexicons.py


# ===============================
#  Helper สำหรับทำความสะอาด / เช็คคำ
# ===============================

def _normalize_text(s: str) -> str:
    """แปลงเป็นตัวพิมพ์เล็ก + รวม space ให้เหลือช่องเดียว"""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _contains_any(text, patterns):
    for pat in patterns:
        # match เฉพาะ whole word เช่น sex → ต้องยืนเดี่ยว
        if re.search(rf"\b{re.escape(pat)}\b", text):
            return True
    return False


# ===============================================
#  Rule-based tagging ต่อ segment (row ใน DataFrame)
# ===============================================

def apply_rule_based_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มคอลัมน์ rule-based flags:
      - profanity, sexual, violence, hate (0/1)
      - profanity_hybrid, sexual_hybrid, ... (ตั้ง = flag เดียวกัน)
      - has_* flags (strong / mild / slur) ต่อ segment
      - *_prob = 0.0 (เพราะไม่มีโมเดล)
    """
    texts_norm = df["text"].astype(str).apply(_normalize_text)

    prof_flags, sex_flags, viol_flags, hate_flags = [], [], [], []
    has_prof_strong_list, has_sex_strong_list = [], []
    has_sex_mild_list, has_viol_strong_list = [], []
    has_viol_mild_list, has_hate_slur_list = [], []

    for t in texts_norm:
        # profanity
        has_prof_strong = _contains_any(t, lexicons.PROFANITY_STRONG)
        has_prof_mild   = _contains_any(t, lexicons.PROFANITY_MILD)
        prof = 1 if (has_prof_strong or has_prof_mild) else 0

        # sexual
        has_sex_strong = _contains_any(t, lexicons.SEXUAL_STRONG)
        has_sex_mild   = _contains_any(t, lexicons.SEXUAL_MILD)
        sex = 1 if (has_sex_strong or has_sex_mild) else 0

        # violence
        has_viol_strong = _contains_any(t, lexicons.VIOLENT_STRONG)
        has_viol_mild   = _contains_any(t, lexicons.VIOLENT_MILD)
        viol = 1 if (has_viol_strong or has_viol_mild) else 0

        # hate / slur
        has_hate_slur = _contains_any(t, lexicons.HATE_SLURS)
        hate = 1 if has_hate_slur else 0

        # append flags
        prof_flags.append(prof)
        sex_flags.append(sex)
        viol_flags.append(viol)
        hate_flags.append(hate)

        has_prof_strong_list.append(int(has_prof_strong))
        has_sex_strong_list.append(int(has_sex_strong))
        has_sex_mild_list.append(int(has_sex_mild))
        has_viol_strong_list.append(int(has_viol_strong))
        has_viol_mild_list.append(int(has_viol_mild))
        has_hate_slur_list.append(int(has_hate_slur))

    out = df.copy()

    # basic flags
    out["profanity"] = prof_flags
    out["sexual"] = sex_flags
    out["violence"] = viol_flags
    out["hate"] = hate_flags

    # hybrid = rule สำหรับ rule-based-only (จะให้ rate_video ใช้คอลัมน์ *_hybrid ได้)
    out["profanity_hybrid"] = out["profanity"]
    out["sexual_hybrid"] = out["sexual"]
    out["violence_hybrid"] = out["violence"]
    out["hate_hybrid"] = out["hate"]

    # strong/mild flags (ใช้ใน rate_video)
    out["has_prof_strong"] = has_prof_strong_list
    out["has_sex_strong"] = has_sex_strong_list
    out["has_sex_mild"] = has_sex_mild_list
    out["has_viol_strong"] = has_viol_strong_list
    out["has_viol_mild"] = has_viol_mild_list
    out["has_hate_slur"] = has_hate_slur_list

    # probability = 0.0 (เพราะเราไม่ได้ใช้โมเดล)
    out["profanity_prob"] = 0.0
    out["sexual_prob"] = 0.0
    out["violence_prob"] = 0.0
    out["hate_prob"] = 0.0

    return out


# ===============================================
#  เขียน TXT output (เหมือน detector ตัวอื่น ๆ)
# ===============================================

def write_flagged_txt_with_rating(flag_csv_path, rating_result, out_txt_path, input_stem: str):
    df = pd.read_csv(flag_csv_path).fillna("")

    rating, reason, detailed_reason, debug = rating_result

    total_flagged = len(df)  # 👈 นับจำนวน segment ที่โดน flag แล้ว

    lines = []
    lines.append("=========== RULE-BASED ONLY DETECTION ==========")
    lines.append(f"INPUT FILE: {input_stem}")
    lines.append(f"OVERALL RATING: {rating}")
    lines.append(f"REASON: {reason}")
    lines.append(f"TOTAL FLAGGED SEGMENTS: {total_flagged}")  # 👈 เพิ่มบรรทัดนี้
    lines.append("-----------------------------------------------")
    lines.append("=========== FLAGGED SEGMENTS ==================")

    for _, row in df.iterrows():
        tags = []
        if row.get("profanity", 0) == 1:
            tags.append("PROFANITY")
        if row.get("sexual", 0) == 1:
            tags.append("SEXUAL")
        if row.get("violence", 0) == 1:
            tags.append("VIOLENCE")
        if row.get("hate", 0) == 1:
            tags.append("HATE")

        raw_start = row.get("start_time", row.get("start", 0))
        raw_end = row.get("end_time", row.get("end", 0))

        sec_start = _to_seconds(raw_start)
        sec_end = _to_seconds(raw_end)

        start = seconds_to_hms(sec_start)
        end = seconds_to_hms(sec_end)

        text = str(row.get("text", "")).strip()
        lines.append(f"[{start} - {end}] [{', '.join(tags)}] {text}")

    lines.append("================================================")

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    #print(f"✅ Saved flagged txt: {out_txt_path}")



# ===============================================
#                   MAIN
# ===============================================

def main():
    ap = argparse.ArgumentParser(description="Rule-based toxicity detection")
    ap.add_argument("--input", help="raw transcript .txt file (full pipeline mode)")
    ap.add_argument("--csv", help="CSV file with 'text' column (inference-only mode)")
    ap.add_argument("--out", help="Output CSV file (required for --csv mode)")
    args = ap.parse_args()

    # ===== MODE 1: CSV input (inference only - for evaluation) =====
    if args.csv:
        if not args.out:
            raise ValueError("--out is required when using --csv mode")

        print(f"📥 Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv).fillna("")

        if 'text' not in df.columns:
            raise ValueError("CSV must have 'text' column")

        print(f"✅ Loaded {len(df)} samples")
        print("🔍 Running rule-based detection...")

        # Apply rule-based tagging
        df_rule = apply_rule_based_tags(df)

        # Save output (only predictions, no rating)
        df_rule.to_csv(args.out, index=False, encoding="utf-8-sig")

        # Print detection statistics
        print(f"\n📊 Detection Results:")
        print(f"  Profanity: {df_rule['profanity'].sum()}/{len(df_rule)} ({df_rule['profanity'].sum()/len(df_rule)*100:.1f}%)")
        print(f"  Sexual:    {df_rule['sexual'].sum()}/{len(df_rule)} ({df_rule['sexual'].sum()/len(df_rule)*100:.1f}%)")
        print(f"  Violence:  {df_rule['violence'].sum()}/{len(df_rule)} ({df_rule['violence'].sum()/len(df_rule)*100:.1f}%)")
        print(f"  Hate:      {df_rule['hate'].sum()}/{len(df_rule)} ({df_rule['hate'].sum()/len(df_rule)*100:.1f}%)")
        print(f"\n✅ Saved: {args.out}")
        return

    # ===== MODE 2: TXT input (full pipeline with preprocessing and rating) =====
    if not args.input:
        raise ValueError("Either --input or --csv must be specified")

    input_txt = Path(args.input)

    # STEP 1: preprocessing (txt -> CSV)
    PRE_CSV = "outputs/transcripts/transcript_clean.csv"
    subprocess.run(
        ["python", "preprocessing.py", "--input", str(input_txt), "--out", PRE_CSV],
        check=True,
    )

    # STEP 2: rule-based tagging on CSV
    df = pd.read_csv(PRE_CSV).fillna("")

    # เดา column name ให้ยืดหยุ่น
    lower_cols = {c.lower(): c for c in df.columns}
    text_col = lower_cols.get("text", None) or lower_cols.get("transcript", None)
    if text_col is None:
        raise ValueError("CSV ต้องมีคอลัมน์ 'text' หรือ 'transcript'")

    start_col = None
    end_col = None
    for cand in ["start_time", "start", "from"]:
        if cand in lower_cols:
            start_col = lower_cols[cand]
            break
    for cand in ["end_time", "end", "to"]:
        if cand in lower_cols:
            end_col = lower_cols[cand]
            break

    if start_col is None or end_col is None:
        raise ValueError("CSV ต้องมีคอลัมน์เวลา เช่น start_time/end_time หรือ start/end")

    # rename ให้เป็นมาตรฐาน
    df = df.rename(
        columns={
            text_col: "text",
            start_col: "start_time",
            end_col: "end_time",
        }
    )

    df_rule = apply_rule_based_tags(df)

    out_dir = Path("outputs/rule_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / f"{input_txt.stem}_rule_full.csv"
    flag_csv = out_dir / f"{input_txt.stem}_rule_flagged.csv"

    df_rule.to_csv(full_csv, index=False, encoding="utf-8-sig")

    df_flag = df_rule[
        (df_rule["profanity"] == 1)
        | (df_rule["sexual"] == 1)
        | (df_rule["violence"] == 1)
        | (df_rule["hate"] == 1)
    ]
    df_flag.to_csv(flag_csv, index=False, encoding="utf-8-sig")

    # STEP 3: คำนวณ rating จาก rule_full.csv
    rating_result = rating_utils.rate_video(str(full_csv))
    rating, reason, detailed_reason, debug = rating_result

    # STEP 4: เขียน flagged.txt + rating
    out_txt = out_dir / f"{input_txt.stem}_rule_flagged.txt"
    write_flagged_txt_with_rating(str(flag_csv), rating_result, str(out_txt), input_txt.stem)

    # STEP 5: สรุปแบบกระชับ
    print("======================================")
    print(f"🎬 INPUT FILE     : {input_txt.name}")
    #print(f"⭐ FINAL RATING   : {rating}")
    print(f"🔢 FLAGGED COUNT  : {len(df_flag)}")
    print(f"📄 CLEAN CSV      : {PRE_CSV}")
    print(f"📄 FULL CSV       : {full_csv}")
    print(f"📄 FLAGGED CSV    : {flag_csv}")
    print(f"📝 RATING OUTPUT  : {out_txt}")

    print("======================================")


if __name__ == "__main__":
    main()
