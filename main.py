# main.py
import os
import subprocess
from hybrid_detector import apply_hybrid_to_csv
from rating_utils import rate_video

# ---------- CONFIG ----------

MODEL_DIR   = r"tox_ft\best_model"   # หรือ tox_ft_pos\best_model แล้วแต่ของน้อง
#CSV_INPUT   = r'Transcript\STOP BEING A PEOPLE PLEASER - Gary Vaynerchuk Motivation.csv'
CSV_INPUT   = r'outputs\transcripts\transcript_clean.csv'
CSV_SCORES  = r"outputs\ver2_with_scores.csv"
CSV_HYBRID  = r"outputs\ver2_with_scores_hybrid.csv"
FLAG_OUTPUT = r"outputs\ver2_flagged.csv"

CONTEXT_WIN = "0"
RATING_OUT  = r"outputs\ver2_video_rating.txt"

# ----------------------------

def main():
    # 0) ตรวจ path
    if not os.path.exists(CSV_INPUT):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ CSV input: {CSV_INPUT}")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์โมเดล: {MODEL_DIR}")

    os.makedirs("outputs", exist_ok=True)

    print("\n🚀 STEP 1: รันโมเดลตรวจ Toxicity (infer.py)\n")

    cmd = [
        "python",
        "model\\infer.py",
        "--model_dir", MODEL_DIR,
        "--csv", CSV_INPUT,
        "--out", CSV_SCORES,
        "--flag_out", FLAG_OUTPUT,
        "--context_window", CONTEXT_WIN,
    ]

    print("📌 Command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    subprocess.run(cmd, check=True)

    print("\n✅ เสร็จ infer_v2")
    print(f"📄 Full scores: {CSV_SCORES}")
    print(f"🚩 Flagged segments: {FLAG_OUTPUT}")

    print("\n🚀 STEP 2: ทำ Hybrid (ML + Lexicon)\n")
    apply_hybrid_to_csv(CSV_SCORES, CSV_HYBRID)

    print("\n🚀 STEP 3: วิเคราะห์เรท 13+/15+/18+/20+\n")
    rating, info , debug = rate_video(CSV_HYBRID)

    print("======================================")
    print(f"⭐ FINAL RATING: {rating}")
    print(f"ℹ️  Reason: {info['reason']}")
    print(f"   r_prof = {info['r_prof']:.4f}")
    print(f"   r_sex  = {info['r_sex']:.4f}")
    print(f"   r_viol = {info['r_viol']:.4f}")
    print(f"   r_hate = {info['r_hate']:.4f}")
    print("======================================")

    with open(RATING_OUT, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {CSV_INPUT}\n")
        f.write(f"MODEL: {MODEL_DIR}\n")
        f.write(f"RATING: {rating}\n")
        f.write(f"REASON: {info['reason']}\n")
        f.write(f"r_prof: {info['r_prof']:.4f}\n")
        f.write(f"r_sex:  {info['r_sex']:.4f}\n")
        f.write(f"r_viol: {info['r_viol']:.4f}\n")
        f.write(f"r_hate: {info['r_hate']:.4f}\n")

        f.write("\n\n=========== DEBUG INFORMATION ===========\n")
        for k, v in debug.items():
            f.write(f"{k}: {v}\n")
        f.write("=========================================\n")

    print(f"\n💾 Rating saved to: {RATING_OUT}")
    print("\n🎉 ALL DONE\n")


if __name__ == "__main__":
    main()
