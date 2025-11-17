# hybrid_detector.py
# รวม logic hybrid ทั้งหมด
# ไฟล์นี้ใช้สร้าง hybrid label (probability จากโมเดล + lexicon-based rules)

import re
import pandas as pd

# ---------- 1) Lexicon sets ----------

PROFANITY_MILD = {
    "damn", "crap", "hell", "shit", "fuck"
}

# เพิ่มคำหยาบแรง ๆ ที่เจอบ่อย
PROFANITY_STRONG = {
    "bitch", "asshole", "dickhead", "cunt", "fucking", "motherfucker"
}

# sexual ที่ explicit → ใช้เป็นตัวแทน strong / 20+
SEXUAL_STRONG_20 = {
    "porn", "porno", "pornography", "xxx",
    "blowjob", "handjob",
    "pussy", "tits", "boobs", "boob",
    "dick", "cock",
    "cum", "orgasm",
    "breasts", "naked",
}

# sexual แบบ context (คำว่า sex, sexual ฯลฯ) — ไม่ใส่ "fuck" ตรงนี้
SEXUAL_MILD = {
    "sex", "sexual", "adult", "pleasure",
}

# violence lexicon เดิม (โอเคแล้ว)
VIOLENCE_STRONG_20 = {
    "kill", "murder", "stab", "torture", "rape",
    "behead", "slaughter", "chop", "execute",
}

VIOLENCE_MILD = {
    "fight", "hit", "beat", "punch",
}

# hate slur แบบตรง ๆ
HATE_SLURS = {
    "nigga", "nigger",
    "faggot", "fag",
    "chink", "spic", "wetback", "kike",
}

# ---------- 2) Decensor + tokenize ----------

DECENSOR_PATTERNS = [
    (r"f[\*\-_\.\s]*u[\*\-_\.\s]*c[\*\-_\.\s]*k", "fuck"),
    (r"s[\*\-_\.\s]*h[\*\-_\.\s]*i[\*\-_\.\s]*t", "shit"),
    (r"b[\*\-_\.\s]*i[\*\-_\.\s]*t[\*\-_\.\s]*c[\*\-_\.\s]*h", "bitch"),
    (r"a[\*\-_\.\s]*s[\*\-_\.\s]*s[\*\-_\.\s]*h[\*\-_\.\s]*o[\*\-_\.\s]*l[\*\-_\.\s]*e", "asshole"),
]


def clean_and_tokenize(text: str):
    s = str(text).lower()
    # decensor
    for pat, rep in DECENSOR_PATTERNS:
        s = re.sub(pat, rep, s)

    # ตัดพวกเครื่องหมายง่าย ๆ
    s = re.sub(r"[\"',!?()\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    return s, tokens


# ---------- 3) Hybrid label สำหรับ 1 แถว ----------

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
    SEX_ACTION_WORDS = {"have", "had", "having", "do", "did", "doing"}
    SEX_PRONOUNS = {"i", "you", "we", "he", "she", "they"}
    # time words ที่บอกบริบท sexual (ไม่ใส่ today เพราะจะทำให้ "sex education in school today" ติด)
    SEX_TIME_WORDS = {"last", "tonight", "yesterday"}

    # 1) ดึง prob จากโมเดล
    p_prof = float(row.get("profanity_prob", 0.0))
    p_sex = float(row.get("sexual_prob", 0.0))
    p_viol = float(row.get("violence_prob", 0.0))
    p_hate = float(row.get("hate_prob", 0.0))

    # 2) Lexicon flags (นับด้วย set + counter)

    # profanity strong / mild
    has_prof_mild = len(token_set & PROFANITY_MILD) > 0
    has_prof_strong = len(token_set & PROFANITY_STRONG) > 0

    # sexual strong / mild
    sex_count_strong = sum(1 for t in tokens if t in SEXUAL_STRONG_20)
    sex_count_mild = sum(1 for t in tokens if t in SEXUAL_MILD)

    has_sex_strong = sex_count_strong >= 1

    # เช็ค context สำหรับ sex แบบ mild (กันเคส "sex education")
    has_sex_word = sex_count_mild >= 1
    has_sex_action_word = any(t in SEX_ACTION_WORDS for t in tokens)
    has_sex_pronoun = any(t in SEX_PRONOUNS for t in tokens)
    has_sex_time_word = any(t in SEX_TIME_WORDS for t in tokens)

    # mild sexual ต้องมี sex + (action/pronoun/time)
    has_sex_mild = has_sex_word and (
        has_sex_action_word or has_sex_pronoun or has_sex_time_word
    )

    # violence strong / mild
    viol_count_strong = sum(1 for t in tokens if t in VIOLENCE_STRONG_20)
    viol_count_mild = sum(1 for t in tokens if t in VIOLENCE_MILD)

    has_viol_strong = viol_count_strong >= 1
    has_viol_mild = viol_count_mild >= 2

    # hate slur แบบตรงตัว
    has_hate_slur = any(t in HATE_SLURS for t in tokens)

    # ---------- 3) Hybrid rules ----------

    # ใช้ lexicon ดัน profanity ให้แน่น
    profanity_hybrid = int(
        (p_prof >= thr_prof)
        or has_prof_strong
        or (has_prof_mild and p_prof >= 0.30)
    )

    # sexual: ใช้ lexicon เป็นหลัก ถ้าไม่มีค่อยให้โมเดลช่วย
    has_any_sex_lex = has_sex_strong or has_sex_mild

    if has_any_sex_lex:
        # มีคำ sexual อยู่แล้ว → ถือว่า sexual แม้ prob จะต่ำ
        sexual_hybrid = 1
    else:
        # ไม่มีคำ sexual ในประโยคเลย
        # ให้โมเดลตัดสินได้เฉพาะตอนที่มั่นใจมาก และต้องไม่ใช่ profanity นำ
        sexual_hybrid = int((p_sex >= thr_sex) and (p_prof < thr_prof))

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


# ---------- 4) ใช้กับทั้ง CSV ----------

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




'''# hybrid_detector.py
# รวม logic hybrid ทั้งหมด
# ไฟล์นี้ใช้สร้าง hybrid label (probability จากโมเดล + lexicon-based rules)
import re
import pandas as pd

# ---------- 1) Lexicon sets ----------

PROFANITY_MILD = {
    "damn", "crap", "hell", "shit","fuck"
}

# เพิ่มคำหยาบแรง ๆ ที่เจอบ่อย
PROFANITY_STRONG = {
    "bitch", "asshole", "dickhead", "cunt", "fucking", "motherfucker"
}

# sexual ที่ explicit → ใช้เป็นตัวแทน strong / 20+
SEXUAL_STRONG_20 = {
    "porn", "porno", "pornography", "xxx",
    "blowjob", "handjob",
    "pussy", "tits", "boobs", "boob",
    "dick", "cock",
    "cum", "orgasm",
    "breasts", "naked"
}


# sexual แบบ context (คำว่า sex, sexual ฯลฯ)
SEXUAL_MILD = {
    "sex", "sexual", "adult", "pleasure"
}

# violence lexicon เดิม (โอเคแล้ว)
VIOLENCE_STRONG_20 = {
    "kill", "murder", "stab", "torture", "rape",
    "behead", "slaughter", "chop", "execute"
}

VIOLENCE_MILD = {
    "fight", "hit", "beat", "punch"
}

# แทนที่จะใช้ MASK แปลก ๆ ให้ใช้ slur ตรง ๆ เลย
HATE_SLURS = {
    "nigga", "nigger",
    "faggot", "fag",
    "chink", "spic", "wetback", "kike"
}

# ---------- 2) Decensor + tokenize ----------

DECENSOR_PATTERNS = [
    (r"f[\*\-_\.\s]*u[\*\-_\.\s]*c[\*\-_\.\s]*k", "fuck"),
    (r"s[\*\-_\.\s]*h[\*\-_\.\s]*i[\*\-_\.\s]*t", "shit"),
    (r"b[\*\-_\.\s]*i[\*\-_\.\s]*t[\*\-_\.\s]*c[\*\-_\.\s]*h", "bitch"),
    (r"a[\*\-_\.\s]*s[\*\-_\.\s]*s[\*\-_\.\s]*h[\*\-_\.\s]*o[\*\-_\.\s]*l[\*\-_\.\s]*e", "asshole"),
    
]

def clean_and_tokenize(text: str):
    s = str(text).lower()
    # decensor
    for pat, rep in DECENSOR_PATTERNS:
        s = re.sub(pat, rep, s)

    # ตัดพวกเครื่องหมายง่าย ๆ
    s = re.sub(r"[\"',!?()\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    return s, tokens


# ---------- 3) Hybrid label สำหรับ 1 แถว ----------

def hybrid_labels_for_row(row,
                          thr_prof=0.45,
                          thr_sex=0.55,
                          thr_viol=0.03,
                          thr_hate=0.5):
    text = row["text"]
    s, tokens = clean_and_tokenize(text)
    token_set = set(tokens)

    # คำที่บ่งบอกว่า sex เป็น action จริง ๆ (ไม่ใช่วิชาการ)
    SEX_ACTION_WORDS = {
        "have", "had", "having",
        "do", "did", "doing",
        "fuck", "fucking"
    }
    
    SEX_TIME_WORDS = {
        "last", "tonight", "yesterday", "today"
    }
  
    SEX_PRONOUNS = {
        "i", "you", "we", "he", "she", "they"
    }

    # 1) ดึง prob จากโมเดล
    p_prof = float(row.get("profanity_prob", 0.0))
    p_sex  = float(row.get("sexual_prob", 0.0))
    p_viol = float(row.get("violence_prob", 0.0))
    p_hate = float(row.get("hate_prob", 0.0))

    # 2) Lexicon flags (นับด้วย set + counter)

    # profanity strong / mild
    has_prof_mild   = len(token_set & PROFANITY_MILD)   > 0
    has_prof_strong = len(token_set & PROFANITY_STRONG) > 0

    # ---- SEXUAL LOGIC (FIXED) ----

    SEX_ACTION_WORDS = {"have", "had", "having", "do", "did", "doing"}
    SEX_PRONOUNS = {"i", "you", "we", "he", "she", "they"}
    SEX_TIME_WORDS = {"last", "tonight", "yesterday"}

    sex_count_strong = sum(1 for t in tokens if t in SEXUAL_STRONG_20)
    sex_count_mild   = sum(1 for t in tokens if t in SEXUAL_MILD)

    has_sex_strong = sex_count_strong >= 1

    has_sex_word = sex_count_mild >= 1
    has_sex_action_word = any(t in SEX_ACTION_WORDS for t in tokens)
    has_sex_pronoun = any(t in SEX_PRONOUNS for t in tokens)
    has_sex_time_word = any(t in SEX_TIME_WORDS for t in tokens)

    # mild sexual ต้องมี sex + (action/pronoun/time)
    has_sex_mild = has_sex_word and (has_sex_action_word or has_sex_pronoun or has_sex_time_word)

    # ---- HYBRID RULE FOR SEXUAL ----

    has_any_sex_lex = has_sex_strong or has_sex_mild

    if has_any_sex_lex:
        sexual_hybrid = 1
    else:
        # ให้โมเดลตัดสินได้เฉพาะตอนที่มั่นใจมาก และต้องไม่ใช่ profanity
        sexual_hybrid = int((p_sex >= 0.90) and (p_prof < 0.50))


    # violence strong / mild
    viol_count_strong = sum(1 for t in tokens if t in VIOLENCE_STRONG_20)
    viol_count_mild   = sum(1 for t in tokens if t in VIOLENCE_MILD)

    has_viol_strong = viol_count_strong >= 1
    has_viol_mild   = viol_count_mild   >= 2

    # hate slur แบบตรงตัว
    has_hate_slur = any(t in HATE_SLURS for t in tokens)

    # 3) Hybrid rules

    # ใช้ lexicon ดัน profanity ให้แน่น
    profanity_hybrid = int(
        (p_prof >= thr_prof) or
        has_prof_strong or
        (has_prof_mild and p_prof >= 0.3)
    )

    # ถ้ามี lexicon sexual → ใช้เป็นหลัก
    has_any_sex_lex = has_sex_strong or has_sex_mild

    if has_any_sex_lex:
        # มีคำ sexual อยู่แล้ว → ถือว่า sexual แม้ prob จะต่ำ
        sexual_hybrid = 1
    else:
        # ไม่มีคำ sexual ในประโยคเลย
        # อนุญาตให้ model ฟันว่า sexual ได้ก็ต่อเมื่อมั่นใจมาก และไม่ได้เป็น profanity นำ
        sexual_hybrid = int(
            (p_sex >= thr_sex)
        )

    violence_hybrid = int(
        (p_viol >= thr_viol) or
        has_viol_strong or
        has_viol_mild
    )

    hate_hybrid = int(
        (p_hate >= thr_hate) or
        has_hate_slur
    )

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


# ---------- 4) ใช้กับทั้ง CSV ----------

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
'''