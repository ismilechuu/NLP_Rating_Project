# hybrid_detector.py
# รวม logic hybrid ทั้งหมด
# ไฟล์นี้ใช้สร้าง hybrid label (probability จากโมเดล + lexicon-based rules)

import re
import pandas as pd
from lexicons import (
    PROFANITY_MILD,
    PROFANITY_STRONG,
    SEXUAL_STRONG,
    SEXUAL_MILD,  
    SEX_EDU_WHITELIST,
    VIOLENT_MILD ,
    VIOLENT_STRONG,
    HATE_SLURS,
    
)

def clean_and_tokenize(text: str):
    s = str(text).lower()

    # ตัดพวกเครื่องหมายง่าย ๆ
    s = re.sub(r"[\"',!?()\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    return s, tokens


# ---------- Hybrid label สำหรับ 1 แถว ----------

def hybrid_labels_for_row(
    row,
    thr_prof: float = 0.45,
    thr_sex: float = 0.90,
    thr_viol: float = 0.03,
    thr_hate: float = 0.50,
):
    text = row["text"]
    s, tokens = clean_and_tokenize(text)
    token_set = set(tokens)

    # คำที่บ่งบอกว่า sex เป็น action จริง ๆ (ไม่ใช่วิชาการ)
    SEX_ACTION_WORDS = {"have", "had", "having", "do", "did", "doing", "going", "was", "were"}
    #SEX_PRONOUNS = {"i", "you", "we", "he", "she", "they"}
    #SEX_TIME_WORDS = {"last", "tonight", "yesterday"}

    # 1) ดึง prob จากโมเดล
    p_prof = float(row.get("profanity_prob", 0.0))
    p_sex = float(row.get("sexual_prob", 0.0))
    p_viol = float(row.get("violence_prob", 0.0))
    p_hate = float(row.get("hate_prob", 0.0))

    # 2) Lexicon flags (นับด้วย set + counter)

    # profanity strong / mild
    has_prof_mild = len(token_set & PROFANITY_MILD) > 0
    has_prof_strong = len(token_set & PROFANITY_STRONG) > 0

    # sexual logic
    clean = s  # clean string จาก clean_and_tokenize
    is_edu_sex = any(phrase in clean for phrase in SEX_EDU_WHITELIST)
    sex_count_strong = sum(1 for t in tokens if t in SEXUAL_STRONG)
    has_sex_strong = sex_count_strong >= 1

    sex_count_mild = sum(1 for t in tokens if t in SEXUAL_MILD)
    has_sex_word = sex_count_mild >= 1

    has_sex_action_word = any(t in SEX_ACTION_WORDS for t in tokens)
    has_sex_mild = has_sex_word and has_sex_action_word


    # violence strong / mild
    viol_count_strong = sum(1 for t in tokens if t in VIOLENT_STRONG)
    viol_count_mild = sum(1 for t in tokens if t in VIOLENT_MILD)

    has_viol_strong = viol_count_strong >= 1
    has_viol_mild = viol_count_mild >= 2

    # hate slur แบบตรงตัว
    has_hate_slur = any(t in HATE_SLURS for t in tokens)

    # ---------- 3) Hybrid rules ----------

    # ใช้ lexicon ดัน profanity ให้แน่น
    profanity_hybrid = int(
        (p_prof >= thr_prof)
        or has_prof_strong
        or (has_prof_mild and p_prof >= 0.08)
    )

    # sexual: ใช้ lexicon เป็นหลัก ถ้าไม่มีค่อยให้โมเดลช่วย
    has_any_sex_lex = has_sex_strong or has_sex_mild

    if is_edu_sex:
    # เคส educational → ไม่ใช่ sexual แน่นอน
        sexual_hybrid = 0
    elif has_any_sex_lex:
        # มีคำ sexual อยู่จริง (strong/mild) → sexual แน่นอน
        sexual_hybrid = 1
    else:
        # ใช้โมเดลตัดสินเฉพาะคำที่ model มั่นใจมาก
        sexual_hybrid = int(p_sex >= thr_sex) #and (p_prof < thr_prof)

    violence_hybrid = int(
        (p_viol >= thr_viol) or has_viol_strong or has_viol_mild
    )

    hate_hybrid = int((p_hate >= thr_hate) or has_hate_slur)

    return {
        "profanity_hybrid": profanity_hybrid,
        "sexual_hybrid": sexual_hybrid,
        "violence_hybrid": violence_hybrid,
        "hate_hybrid": hate_hybrid,
        "has_prof_strong": int(has_prof_strong),
        "has_sex_strong": int(has_sex_strong),
        "has_sex_mild": int(has_sex_mild),
        "has_violence_strong": int(has_viol_strong),
        "has_violence_mild": int(has_viol_mild),
        "has_hate_slur": int(has_hate_slur),
    }


# ----------  ใช้กับทั้ง CSV ----------

def apply_hybrid_to_csv(csv_in: str, csv_out: str):
    df = pd.read_csv(csv_in)

    hybrid_rows = []
        
    for _, row in df.iterrows():
        hybrid = hybrid_labels_for_row(row)
        hybrid_rows.append(hybrid)

    hybrid_df = pd.DataFrame(hybrid_rows)
    out_df = pd.concat([df, hybrid_df], axis=1)

    out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"✅ Saved hybrid result to: {csv_out}")


if __name__ == "__main__":
    # เผื่ออยากรันไฟล์นี้เดี่ยว ๆ
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--csv_out", required=True)
    args = ap.parse_args()

    apply_hybrid_to_csv(args.csv_in, args.csv_out)
