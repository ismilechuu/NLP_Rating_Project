# rating_utils.py
# ไฟล์นี้ใช้ อ่าน CSV (ที่ hybrid แล้ว) แล้วสรุปเรต 13+/15+/18+/20+
# ใช้ lexicon-based rules ร่วมกับค่า probability จากโมเดล
import re
import pandas as pd
from hybrid_detector import PROFANITY_MILD, PROFANITY_STRONG, SEXUAL_STRONG_20, SEXUAL_MILD, VIOLENCE_STRONG_20, VIOLENCE_MILD

# สร้าง set ของ token จาก dataframe
# โดยเอาข้อความทุกแถวมาต่อกัน → ทำ token set หนึ่งชุด
def _token_set_from_df(df: pd.DataFrame):
    text_all = " ".join(df["text"].astype(str).tolist()).lower()
    text_all = re.sub(r"[\"',!?()\[\]]", " ", text_all)
    text_all = re.sub(r"\s+", " ", text_all).strip()
    return set(text_all.split())

# ฟังก์ชันหลัก: rate video จาก CSV path
# อินพุต = CSV ที่ hybrid แล้ว (ver2_with_scores_hybrid.csv)
# คืนค่า (rating_str, info_dict, debug_dict)
#
def rate_video(csv_path: str):
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        return "UNKNOWN", {"reason": "no segments"}

    # ใช้ hybrid ถ้ามี ไม่มีก็ fallback
    def col(name, fallback):
        if name in df.columns:
            return df[name]
        return df[fallback]

    prof_col = col("profanity_hybrid", "profanity")
    sex_col  = col("sexual_hybrid", "sexual")
    viol_col = col("violence_hybrid", "violence")
    hate_col = col("hate_hybrid", "hate")

    # อัตราส่วน segment ที่โดนแต่ละหมวด
    r_prof = prof_col.mean()
    r_sex  = sex_col.mean()
    r_viol = viol_col.mean()
    r_hate = hate_col.mean()

    # จำนวน segment ที่โดนจริง ๆ
    n_seg   = len(df)
    n_prof  = int(prof_col.sum())
    n_sex   = int(sex_col.sum())
    n_viol  = int(viol_col.sum())
    n_hate  = int(hate_col.sum())

    # max prob เผื่อใช้ใน rule
    max_prof = df.get("profanity_prob", pd.Series([0])).max()
    max_sex  = df.get("sexual_prob", pd.Series([0])).max()
    max_viol = df.get("violence_prob", pd.Series([0])).max()
    max_hate = df.get("hate_prob", pd.Series([0])).max()

    tokens = _token_set_from_df(df)

    has_prof_strong = bool(df.get("has_prof_strong", pd.Series([0])).max())
    has_sex_strong  = bool(df.get("has_sex_strong",  pd.Series([0])).max())
    has_sex_mild    = bool(df.get("has_sex_mild",    pd.Series([0])).max())
    has_viol_strong = bool(df.get("has_viol_strong", pd.Series([0])).max())
    has_viol_mild   = bool(df.get("has_viol_mild",   pd.Series([0])).max())


    

    # ====== DEBUG INFO รวมทุกอย่างที่ใช้คิดเรท ======
    debug = {
        "n_segments": n_seg,
        "n_prof_segments": n_prof,
        "n_sex_segments": n_sex,
        "n_viol_segments": n_viol,
        "n_hate_segments": n_hate,

        "r_prof": float(r_prof),
        "r_sex": float(r_sex),
        "r_viol": float(r_viol),
        "r_hate": float(r_hate),

        "max_prof_prob": float(max_prof),
        "max_sex_prob": float(max_sex),
        "max_viol_prob": float(max_viol),
        "max_hate_prob": float(max_hate),

        "has_prof_strong": has_prof_strong,
        "has_sex_strong": has_sex_strong,
        "has_sex_mild": has_sex_mild,
        "has_viol_strong": has_viol_strong,
        "has_viol_mild": has_viol_mild,

        # ตัวอย่าง token (เอาไว้ดูคร่าว ๆ)
        "sample_tokens": sorted(list(tokens))[:40],
    }
    '''
    if verbose:
        print("\n===== DEBUG RATING INPUT =====")
        for k, v in debug.items():
            print(f"{k}: {v}")
        print("================================\n")
    '''
    # ---------- RULES ----------'
    
    # 20+ = ต้อง explicit จริง + โผล่พอสมควร
    if (has_sex_strong and r_sex >= 0.20) or (has_viol_strong and r_viol >= 0.20):
        return "20+", {
            "r_prof": float(r_prof),
            "r_sex": float(r_sex),
            "r_viol": float(r_viol),
            "r_hate": float(r_hate),
            "reason": "explicit sexual/violent content (frequent)"
        }, debug

    # 2) 18+ : ภาษาหยาบหนัก / sexual/hate ปานกลาง
    if (
        r_prof >= 0.15 or
        max_prof >= 0.8 or
        r_sex >= 0.05 or
        (r_hate >= 0.02 and max_hate >= 0.6)
    ):
        return "18+", {
            "r_prof": float(r_prof),
            "r_sex": float(r_sex),
            "r_viol": float(r_viol),
            "r_hate": float(r_hate),
            "reason": "strong language and/or moderate sexual/hate content"
        },debug

    # 3) 15+ : คำหยาบมีระดับ (ไม่เบาจน 13+)
    if (
        r_prof >= 0.05 or
        max_prof >= 0.6
    ):
        return "15+", {
            "r_prof": float(r_prof),
            "r_sex": float(r_sex),
            "r_viol": float(r_viol),
            "r_hate": float(r_hate),
            "reason": "moderate profanity"
        },debug

    # 4) 13+ : แทบไม่มีอะไรน่าห่วง
    return "13+", {
        "r_prof": float(r_prof),
        "r_sex": float(r_sex),
        "r_viol": float(r_viol),
        "r_hate": float(r_hate),
        "reason": "minimal bad language"
    },debug

def save_debug(debug_dict, path="outputs/rating_debug.txt"):
    """Save debug info as readable text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== DEBUG RATING INPUT =====\n")
        for k, v in debug_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("================================\n")
    print(f"💾 Debug saved to {path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/video_rating.txt")
    args = ap.parse_args()

    rating, info, debug = rate_video(args.csv)
    print("⭐ RATING:", rating)
    print("ℹ️", info)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("=========== VIDEO RATING RESULT ===========\n")
        f.write(f"TITLE: {args.csv}\n")
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

    print(f"💾 Rating + Debug saved to: {args.out}")
