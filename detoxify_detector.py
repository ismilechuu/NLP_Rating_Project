# detoxify_detector.py
"""
Baseline: Detoxify-only pipeline

การใช้งาน:
    # Full pipeline (with preprocessing and rating)
    python detoxify_detector.py --input path/to/transcript.txt

    # Inference only (for evaluation - skip preprocessing and rating)
    python detoxify_detector.py --csv input.csv --out output.csv

สิ่งที่สคริปต์ทำ:
    1) เรียก preprocessing.py ให้ แปลง .txt -> .csv (clean แล้ว)
    2) รัน Detoxify บน CSV นั้น
    3) สร้างไฟล์ .txt สรุป segment ที่ติดแท็กแบบ:
       [HH:MM:SS - HH:MM:SS] [PROFANITY, SEXUAL] text...
"""

from typing import List, Dict, Any, Optional
import math, re, os, argparse, subprocess
from pathlib import Path

import pandas as pd
from detoxify import Detoxify
import rating_utils


# -------------------- Utils -------------------- #

def _to_seconds(x):
    """รองรับ float/int และสตริง HH:MM:SS(.ms) → วินาที(float)"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if ":" in s:
        h, m, sec = s.split(":")
        return float(int(h) * 3600 + int(m) * 60 + float(sec))
    try:
        return float(s)
    except Exception:
        return None


def seconds_to_hms(sec: Optional[float]) -> str:
    """แปลงวินาที → HH:MM:SS"""
    if sec is None or math.isnan(sec):
        return "00:00:00"
    sec_int = int(sec)
    h = sec_int // 3600
    m = (sec_int % 3600) // 60
    s = sec_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _auto_detect_columns(df: pd.DataFrame):
    """เดาคอลัมน์ text/start/end แบบยืดหยุ่น"""

    lower_cols = {c.lower(): c for c in df.columns}

    text_candidates = ["text", "transcript", "sentence", "content", "dialog", "utterance"]
    start_candidates = ["start", "start_time", "from", "begin", "start time"]
    end_candidates = ["end", "end_time", "to", "finish", "end time"]

    def find_match(candidates):
        for name in candidates:
            if name in lower_cols:
                return lower_cols[name]
        return None

    text_col = find_match(text_candidates)
    start_col = find_match(start_candidates)
    end_col = find_match(end_candidates)

    if text_col is None:
        raise ValueError("CSV ต้องมีคอลัมน์ข้อความ เช่น: text, transcript, sentence, content")
    if start_col is None or end_col is None:
        raise ValueError("CSV ต้องมีคอลัมน์เวลา เช่น start/start_time และ end/end_time")

    return text_col, start_col, end_col


# -------------------- Core Class -------------------- #

class TranscriptModerationSystem:
    """
    Detoxify-only baseline
    """

    def __init__(self, detoxify_variant: str = "original"):
        print(f"Loading Detoxify model ({detoxify_variant})...")
        self.toxicity_model = Detoxify(detoxify_variant)
        print("✅ Detoxify ready.")

    def analyze_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(csv_path)
        text_col, start_col, end_col = _auto_detect_columns(df)
        return self.analyze_dataframe(df, text_col, start_col, end_col)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        start_col: str,
        end_col: str,
    ) -> List[Dict[str, Any]]:
        for c in [text_col, start_col, end_col]:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        starts = df[start_col].apply(_to_seconds).tolist()
        ends = df[end_col].apply(_to_seconds).tolist()
        texts = df[text_col].astype(str).tolist()
        return self._analyze_texts_with_times(texts, starts, ends)

    def _analyze_texts_with_times(
        self,
        texts: List[str],
        starts: List[Optional[float]],
        ends: List[Optional[float]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:

        cleaned = texts   
        out: List[Dict[str, float]] = []

        for i in range(0, len(cleaned), batch_size):
            chunk = cleaned[i:i + batch_size]
            scores = self.toxicity_model.predict(chunk)

            if isinstance(scores, dict):
                keys = list(scores.keys())
                L = len(scores[keys[0]])
                for j in range(L):
                    out.append({k: float(scores[k][j]) for k in keys})
            elif isinstance(scores, list):
                out.extend([{k: float(v) for k, v in d.items()} for d in scores])
            else:
                out.append({k: float(v) for k, v in scores.items()})

        results = []
        for st, en, raw_t, sc in zip(starts, ends, texts, out):
            results.append({
                "start": st,
                "end": en,
                "text": raw_t.strip(),
                "toxicity": float(sc.get("toxicity", 0.0)),
                "severe_toxicity": float(sc.get("severe_toxicity", 0.0)),
                "obscene": float(sc.get("obscene", 0.0)),
                "threat": float(sc.get("threat", 0.0)),
                "insult": float(sc.get("insult", 0.0)),
                "identity_attack": float(sc.get("identity_attack", 0.0)),
            })
        return results

    def to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(results)


# -------------------- Tagging & Report -------------------- #

def get_tags_for_segment(
    item: Dict[str, Any],
    thr_prof: float = 0.5,
    thr_sex: float = 0.5,
    thr_viol: float = 0.8,
    thr_hate: float = 0.5,
    thr_ins: float = 0.8,
) -> List[str]:
    """
    แมป Detoxify → แท็กแบบโปรเจกต์เรา:
      PROFANITY  ~ toxicity/obscene/insult สูง
      SEXUAL     ~ obscene สูง (แบบหยาบ/ติดเรท)
      VIOLENCE   ~ threat สูง
      HATE       ~ identity_attack สูง
    """
    tags = []
    tox = item.get("toxicity", 0.0)
    obsc = item.get("obscene", 0.0)
    ins = item.get("insult", 0.0)
    thr = item.get("threat", 0.0)
    idatk = item.get("identity_attack", 0.0)

    if tox >= thr_prof or obsc >= thr_prof or ins >= thr_prof:
        tags.append("PROFANITY")
    if obsc >= thr_sex:
        tags.append("SEXUAL")
    if thr >= thr_viol:
        tags.append("VIOLENCE")
    if idatk >= thr_hate or ins >= thr_ins:
        tags.append("HATE")

    return tags


def write_flagged_report_txt(
    results: List[Dict[str, Any]],
    out_path: str,
    thr_prof: float = 0.5,
    thr_sex: float = 0.5,
    thr_viol: float = 0.5,
    thr_hate: float = 0.5,
):
    lines = []
    lines.append("=========== FLAGGED SEGMENTS (BY TAG) ==========")

    for item in results:
        tags = get_tags_for_segment(item, thr_prof, thr_sex, thr_viol, thr_hate)
        if not tags:
            continue

        start_str = seconds_to_hms(item.get("start"))
        end_str = seconds_to_hms(item.get("end"))
        tag_str = ", ".join(tags)
        text = item.get("text", "").strip()

        lines.append(f"[{start_str} - {end_str}] [{tag_str}] {text}")

    lines.append("=========================================")

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved flagged report to: {out_path}")


# -------------------- Main CLI -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Detoxify baseline pipeline"
    )
    parser.add_argument("--input", help="raw transcript .txt file (full pipeline mode)")
    parser.add_argument("--csv", help="CSV file with 'text' column (inference-only mode)")
    parser.add_argument("--out", help="Output CSV file (required for --csv mode)")
    parser.add_argument("--csv-out", default="outputs/transcripts/transcript_clean.csv")
    parser.add_argument("--txt-out", default=None)
    parser.add_argument("--model", default="unbiased", choices=["original", "unbiased", "multilingual"],
                        help="Detoxify model variant")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    # ===== MODE 1: CSV input (inference only - for evaluation) =====
    if args.csv:
        if not args.out:
            raise ValueError("--out is required when using --csv mode")

        print(f"📥 Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv).fillna("")

        if 'text' not in df.columns:
            raise ValueError("CSV must have 'text' column")

        print(f"✅ Loaded {len(df)} samples")
        print(f"🤗 Loading Detoxify model: {args.model}")

        # Load Detoxify model
        model = Detoxify(args.model)

        print(f"🔍 Running Detoxify inference (threshold={args.threshold})...")

        # Run inference
        texts = df['text'].tolist()
        all_results = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = model.predict(batch_texts)

            # Process each text in batch
            for j in range(len(batch_texts)):
                text_results = {key: batch_results[key][j] for key in batch_results.keys()}

                # Map Detoxify labels to project labels
                toxicity = text_results.get('toxicity', 0.0)
                obscene = text_results.get('obscene', 0.0)
                insult = text_results.get('insult', 0.0)
                threat = text_results.get('threat', 0.0)
                severe_toxicity = text_results.get('severe_toxicity', 0.0)
                identity_attack = text_results.get('identity_attack', 0.0)
                sexual_explicit = text_results.get('sexual_explicit', 0.0)

                # Map to project labels
                profanity_prob = max(obscene, insult, toxicity)
                sexual_prob = max(sexual_explicit, obscene * 0.5)
                violence_prob = max(threat, severe_toxicity)
                hate_prob = identity_attack

                # Binary predictions
                profanity = 1 if profanity_prob >= args.threshold else 0
                sexual = 1 if sexual_prob >= args.threshold else 0
                violence = 1 if violence_prob >= args.threshold else 0
                hate = 1 if hate_prob >= args.threshold else 0

                all_results.append({
                    'profanity': profanity,
                    'sexual': sexual,
                    'violence': violence,
                    'hate': hate,
                    'profanity_prob': profanity_prob,
                    'sexual_prob': sexual_prob,
                    'violence_prob': violence_prob,
                    'hate_prob': hate_prob,
                    'toxicity': toxicity,
                    'severe_toxicity': severe_toxicity,
                    'obscene': obscene,
                    'threat': threat,
                    'insult': insult,
                    'identity_attack': identity_attack,
                    'sexual_explicit': sexual_explicit,
                })

            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")

        # Create output dataframe
        results_df = pd.DataFrame(all_results)
        output_df = pd.DataFrame({
            'text': df['text'],
            'profanity': results_df['profanity'],
            'sexual': results_df['sexual'],
            'violence': results_df['violence'],
            'hate': results_df['hate'],
        })

        # Add probabilities and raw scores
        for col in ['profanity_prob', 'sexual_prob', 'violence_prob', 'hate_prob',
                    'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult',
                    'identity_attack', 'sexual_explicit']:
            output_df[col] = results_df[col]

        # Save
        output_df.to_csv(args.out, index=False)

        # Print detection statistics
        print(f"\n📊 Detection Results:")
        print(f"  Profanity: {output_df['profanity'].sum()}/{len(output_df)} ({output_df['profanity'].sum()/len(output_df)*100:.1f}%)")
        print(f"  Sexual:    {output_df['sexual'].sum()}/{len(output_df)} ({output_df['sexual'].sum()/len(output_df)*100:.1f}%)")
        print(f"  Violence:  {output_df['violence'].sum()}/{len(output_df)} ({output_df['violence'].sum()/len(output_df)*100:.1f}%)")
        print(f"  Hate:      {output_df['hate'].sum()}/{len(output_df)} ({output_df['hate'].sum()/len(output_df)*100:.1f}%)")
        print(f"\n✅ Saved: {args.out}")
        return

    # ===== MODE 2: TXT input (full pipeline with preprocessing and rating) =====
    if not args.input:
        raise ValueError("Either --input or --csv must be specified")

    input_txt = Path(args.input)
    if not input_txt.exists():
        raise FileNotFoundError(f"Input transcript not found: {input_txt}")

    # Output paths
    csv_out = Path(args.csv_out)
    txt_out = Path(args.txt_out) if args.txt_out else Path("outputs/detoxify") / f"{input_txt.stem}_detoxify_flagged.txt"
    scores_csv = Path("outputs/detoxify") / f"{input_txt.stem}_detoxify_all_segments.csv"

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    scores_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) Preprocessing
    subprocess.run(
        ["python", "preprocessing.py", "--input", str(input_txt), "--out", str(csv_out)],
        check=True,
    )

    # 2) Detoxify inference
    system = TranscriptModerationSystem(detoxify_variant=args.model)
    results = system.analyze_csv(str(csv_out))

    # Save scores CSV
    df_all = system.to_dataframe(results)
    df_all.to_csv(scores_csv, index=False, encoding="utf-8-sig")

    # Write flagged TXT
    write_flagged_report_txt(results, str(txt_out))

        # ----- NEW: Calculate rating -----
    rating, reason, detailed_reason, debug = rating_utils.rate_video(str(scores_csv))

    # นับจำนวน segment ที่โดน flag จาก df_all
    tag_cols = [c for c in ["profanity", "sexual", "violence", "hate"] if c in df_all.columns]
    if tag_cols:
        flagged_mask = (df_all[tag_cols] > 0).any(axis=1)
        flagged_count = int(flagged_mask.sum())
    else:
        flagged_count = 0

    # Summary
    print("======================================")
    print(f"🎬 INPUT FILE     : {input_txt.name}")
    #print(f"⭐ FINAL RATING   : {rating}")
    print(f"🔢 FLAGGED COUNT  : {flagged_count}")
    print(f"📄 CLEAN CSV      : {csv_out}")
    print(f"📄 SCORES CSV     : {scores_csv}")
    print(f"📝 FLAGGED TXT    : {txt_out}")
    print("======================================")




if __name__ == "__main__":
    main()
