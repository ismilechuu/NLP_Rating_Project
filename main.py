# main.py
import os
import argparse
import subprocess

import pandas as pd

from preprocessing import process_file
from hybrid_detector import apply_hybrid_to_csv
from rating_utils import rate_video


def main():
    # ---------- 1) อ่าน arguments ----------
    parser = argparse.ArgumentParser(
        description="Run full toxicity + rating pipeline on a transcript file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="พาธไฟล์ transcript ต้นฉบับ (.txt หรือ .csv) เช่น transcripts/tiktok.txt",
    )
    parser.add_argument(
        "--model_dir",
        "-m",
        default=r"tox_ft\\best_model",
        help="โฟลเดอร์โมเดลที่ใช้ infer (default: tox_ft\\best_model)",
    )
    parser.add_argument(
        "--context_window",
        "-c",
        default="0",
        help="context window สำหรับ infer.py (default: 0)",
    )
    args = parser.parse_args()

    raw_input_path = args.input
    model_dir = args.model_dir
    context_win = str(args.context_window)

    if not os.path.exists(raw_input_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ input: {raw_input_path}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์โมเดล: {model_dir}")

    # ---------- 2) สร้างชื่อไฟล์ตาม base name ----------
    base_name = os.path.splitext(os.path.basename(raw_input_path))[0]

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "transcripts"), exist_ok=True)

    clean_csv   = os.path.join("outputs", "transcripts", f"{base_name}_clean.csv")
    scores_csv  = os.path.join("outputs", f"{base_name}_with_scores.csv")
    flag_csv    = os.path.join("outputs", f"{base_name}_flagged.csv")
    hybrid_csv  = os.path.join("outputs", f"{base_name}_with_scores_hybrid.csv")
    rating_txt  = os.path.join("outputs", f"{base_name}_Rating.txt")

    print("======================================")
    print(f"🎬 INPUT FILE     : {raw_input_path}")
    print(f"📁 MODEL DIR      : {model_dir}")
    print(f"📄 CLEAN CSV      : {clean_csv}")
    print(f"📄 SCORES CSV     : {scores_csv}")
    print(f"📄 FLAGGED CSV    : {flag_csv}")
    print(f"📄 HYBRID CSV     : {hybrid_csv}")
    print(f"📝 RATING OUTPUT  : {rating_txt}")
    print("======================================\n")

    # ---------- STEP 0: Preprocessing ----------
    print("🚀 STEP 0: Preprocessing (TXT/CSV → clean CSV)\n")
    process_file(raw_input_path, clean_csv)

    # ---------- STEP 1: Infer โมเดล ----------
    print("\n🚀 STEP 1: รันโมเดลตรวจ Toxicity (infer.py)\n")

    cmd = [
        "python",
        "model\\infer.py",
        "--model_dir", model_dir,
        "--csv", clean_csv,
        "--out", scores_csv,
        "--flag_out", flag_csv,
        "--context_window", context_win,
    ]

    print("📌 Command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    subprocess.run(cmd, check=True)

    print("\n✅ เสร็จ infer_v2")
    print(f"📄 Full scores: {scores_csv}")
    print(f"🚩 Flagged segments: {flag_csv}")

    # ---------- STEP 2: Hybrid ----------
    print("\n🚀 STEP 2: ทำ Hybrid (ML + Lexicon)\n")
    apply_hybrid_to_csv(scores_csv, hybrid_csv)

    # ---------- STEP 3: Rating ----------
    print("\n🚀 STEP 3: วิเคราะห์เรท 13+/15+/18+/20+\n")
    #rating, info, debug = rate_video(hybrid_csv)
    rating, reason, detailed_reason, debug = rate_video(hybrid_csv)

    print("======================================")
    print(f"⭐ FINAL RATING: {rating}")
    print(f"ℹ️ Reason: {debug.get('reason', '')}")
    print(f"   r_prof = {debug.get('r_prof', 0.0):.4f}")
    print(f"   r_sex  = {debug.get('r_sex', 0.0):.4f}")
    print(f"   r_viol = {debug.get('r_viol', 0.0):.4f}")
    print(f"   r_hate = {debug.get('r_hate', 0.0):.4f}")
    print("======================================")

    # ---------- STEP 4: เขียนไฟล์สรุป Rating + Flagged segments ----------
    df_h = pd.read_csv(hybrid_csv)

    tag_cols = ["profanity_hybrid", "sexual_hybrid", "violence_hybrid", "hate_hybrid"]
    existing_tag_cols = [c for c in tag_cols if c in df_h.columns]
    

    if existing_tag_cols:
        mask_flagged = (df_h[existing_tag_cols] > 0).any(axis=1)
        df_flagged = df_h[mask_flagged].copy()
    else:
        df_flagged = pd.DataFrame()

    with open(rating_txt, "w", encoding="utf-8") as f:
        f.write(f"INPUT FILE: {raw_input_path}\n")
        #f.write(f"MODEL: {model_dir}\n")
        f.write(f"OVERALL RATING: {rating}\n")
        f.write(f"REASON: {reason}\n\n")

        f.write("========== DETAILED REASON ==========\n")
        f.write(detailed_reason + "\n")
        f.write("=====================================\n\n")
        '''
        f.write(f"REASON: {info.get('reason', '')}\n")
        f.write(f"r_prof: {info.get('r_prof', 0.0):.4f}\n")
        f.write(f"r_sex:  {info.get('r_sex', 0.0):.4f}\n")
        f.write(f"r_viol: {info.get('r_viol', 0.0):.4f}\n")
        f.write(f"r_hate: {info.get('r_hate', 0.0):.4f}\n")
        
        f.write("\n\n=========== DEBUG INFORMATION ===========\n")
        for k, v in debug.items():
            f.write(f"{k}: {v}\n")
        f.write("=========================================\n")
        '''
        # แสดงประโยคที่ติดแท็กต่าง ๆ
        if not df_flagged.empty:
            f.write("\n\n=========== FLAGGED SEGMENTS (BY TAG) ==========\n")
            for _, row in df_flagged.iterrows():
                start = str(row.get("start_time", "")).strip()
                end   = str(row.get("end_time", "")).strip()
                text  = str(row.get("text", "")).strip()

                tags = []
                if "profanity_hybrid" in existing_tag_cols and row.get("profanity_hybrid", 0):
                    tags.append("PROFANITY")
                if "sexual_hybrid" in existing_tag_cols and row.get("sexual_hybrid", 0):
                    tags.append("SEXUAL")
                if "violence_hybrid" in existing_tag_cols and row.get("violence_hybrid", 0):
                    tags.append("VIOLENCE")
                if "hate_hybrid" in existing_tag_cols and row.get("hate_hybrid", 0):
                    tags.append("HATE")

                tag_str = ", ".join(tags) if tags else "NONE"
                f.write(f"[{start} - {end}] [{tag_str}] {text}\n")

            f.write("=========================================\n")

    print(f"\n💾 Rating saved to: {rating_txt}")
    print("\n🎉 ALL DONE\n")


if __name__ == "__main__":
    main()
