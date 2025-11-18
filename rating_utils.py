# rating_utils.py
# ไฟล์นี้ใช้ อ่าน CSV (ที่ hybrid แล้ว) แล้วสรุปเรต 13+/15+/18+/20+
# ใช้ lexicon-based rules ร่วมกับค่า probability จากโมเดล

import pandas as pd
import numpy as np

# ========= CONFIG: เกณฑ์เรต (ปรับได้เองตรงนี้) =========

# 20+ : เนื้อหาเพศ/ความรุนแรงค่อนข้างรุนแรงและพบต่อเนื่อง
THR_20_SEX_RATIO   = 0.50   # ถ้า strong sexual >= 50% ของทั้ง transcript → 20+
THR_20_VIOL_RATIO  = 0.50   # ถ้า strong violence >= 50% ของทั้ง transcript → 20+

# 18+ : ด่าหนัก / sexual ปานกลาง / hate speech พอสมควร
THR_18_PROF_RATIO  = 0.2   # สัดส่วน profanity >= 20%
THR_18_PROF_MAX    = 0.80   # หรือเคยด่าหนักมากสักครั้ง (prob >= 0.80)
THR_18_SEX_RATIO   = 0.05   # sexual >= 5% ของทั้ง transcript
THR_18_HATE_RATIO  = 0.02   # hate >= 2%
THR_18_HATE_MAX    = 0.60   # และมี hate_prob สูง >= 0.60

# 15+ : มีคำหยาบเรื่อย ๆ แต่ไม่ถึงขั้น 18+
THR_15_PROF_RATIO  = 0.05   # profanity >= 5%  (อยากเข้ม/ผ่อน ปรับตรงนี้ได้)
THR_15_PROF_MAX    = 0.60   # หรือมีคำหยาบแรงระดับกลางสักครั้ง


# ========= HELPER ฟังก์ชันย่อย =========

def _get_col(df: pd.DataFrame, hybrid: str, model: str) -> pd.Series:
    """
    เลือกใช้คอลัมน์ hybrid ก่อน ถ้าไม่มีค่อย fallback ไปคอลัมน์จากโมเดลโดยตรง
    ถ้าไม่มีทั้งคู่ → คืน Series ที่เป็นศูนย์ทั้งหมด
    """
    if hybrid in df.columns:
        return df[hybrid].fillna(0)
    if model in df.columns:
        return df[model].fillna(0)
    return pd.Series(0, index=df.index, dtype=float)


# ========= ฟังก์ชันหลัก =========

def rate_video(csv_path: str):
    """
    อ่านไฟล์ CSV (หลัง hybrid แล้ว) แล้วสรุปเรตติ้งของทั้ง transcript

    return:
        rating           : "general" / "15+" / "18+" / "20+"
        reason           : เหตุผลสั้น ๆ (สำหรับโชว์บน console / UI)
        detailed_reason  : เหตุผลแบบละเอียด (สำหรับเขียนในไฟล์ Rating.txt)
        debug            : dict ค่าตัวเลขต่าง ๆ เอาไว้ debug / log เพิ่มเติม
    """
    df = pd.read_csv(csv_path)

    # เคสไฟล์ว่าง
    if len(df) == 0:
        rating = "general"
        reason = "ไฟล์ไม่มีข้อมูล ไม่สามารถประเมินเรตได้"
        detailed_reason = (
            "ไฟล์ CSV ไม่มี segment ใด ๆ จึงถือว่าไม่มีเนื้อหาที่ต้องจำกัดเรต "
            "กำหนดเป็นเรต 13+ โดยอัตโนมัติ"
        )
        debug = {"n_segments": 0}
        return rating, reason, detailed_reason, debug

    n_segments = len(df)

    # ใช้ hybrid ก่อน ถ้าไม่มีค่อย fallback ไปคอลัมน์จากโมเดล
    prof = _get_col(df, "profanity_hybrid", "profanity")
    sex  = _get_col(df, "sexual_hybrid", "sexual")
    viol = _get_col(df, "violence_hybrid", "violence")
    hate = _get_col(df, "hate_hybrid", "hate")

    # สัดส่วน segment ที่ถูกแท็กเป็นหมวดต่าง ๆ
    r_prof = float(prof.mean())
    r_sex  = float(sex.mean())
    r_viol = float(viol.mean())
    r_hate = float(hate.mean())

    # ค่า probability สูงสุดจากโมเดล (ถ้ามี)
    max_prof = float(df.get("profanity_prob", pd.Series([0.0])).max())
    max_sex  = float(df.get("sexual_prob",   pd.Series([0.0])).max())
    max_viol = float(df.get("violence_prob", pd.Series([0.0])).max())
    max_hate = float(df.get("hate_prob",     pd.Series([0.0])).max())

    # flag strong/mild จาก hybrid
    has_prof_strong = bool(df.get("has_prof_strong", pd.Series([0])).astype(bool).any())
    has_sex_strong  = bool(df.get("has_sex_strong",  pd.Series([0])).astype(bool).any())
    has_sex_mild    = bool(df.get("has_sex_mild",    pd.Series([0])).astype(bool).any())
    has_viol_strong = bool(df.get("has_viol_strong", pd.Series([0])).astype(bool).any())
    has_viol_mild   = bool(df.get("has_viol_mild",   pd.Series([0])).astype(bool).any())
    has_hate_slur   = bool(df.get("has_hate_slur",   pd.Series([0])).astype(bool).any())

    # ======= เริ่มตัดสินเรต =======
    rating = "general"
    reason = ""
    detailed_reason = ""

    # ---------- 20+ ----------
    if has_sex_strong and r_sex >= THR_20_SEX_RATIO:
        rating = "20+"
        reason = "มีเนื้อหาเชิงเพศรุนแรง (strong sexual) และพบค่อนข้างบ่อย"

        detailed_reason = (
            "จัดเป็นเรต 20+ เพราะมีเนื้อหาเชิงเพศที่รุนแรงและพบต่อเนื่อง\n"
            f"- พบ strong sexual content อย่างน้อย 1 ครั้ง (has_sex_strong = True)\n"
            f"- สัดส่วน segment ที่เป็น sexual = {r_sex:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_20_SEX_RATIO:.2f})\n"
            f"- ค่าความมั่นใจสูงสุดของ sexual จากโมเดล = {max_sex:.2f}\n"
            #"เกณฑ์ที่ใช้: strong sexual + sexual_ratio ≥ THR_20_SEX_RATIO"
        )

    elif has_viol_strong and r_viol >= THR_20_VIOL_RATIO:
        rating = "20+"
        reason = "มีเนื้อหาความรุนแรงรุนแรง (strong violence) และพบค่อนข้างบ่อย"

        detailed_reason = (
            "จัดเป็นเรต 20+ เพราะมีเนื้อหาความรุนแรงที่ชัดเจนและพบต่อเนื่อง\n"
            f"- พบ strong violence content อย่างน้อย 1 ครั้ง (has_viol_strong = True)\n"
            f"- สัดส่วน segment ที่เป็น violence = {r_viol:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_20_VIOL_RATIO:.2f})\n"
            f"- ค่าความมั่นใจสูงสุดของ violence จากโมเดล = {max_viol:.2f}\n"
           # "เกณฑ์ที่ใช้: strong violence + violence_ratio ≥ THR_20_VIOL_RATIO"
        )

    # ---------- 18+ ----------
    elif r_prof >= THR_18_PROF_RATIO:
        rating = "18+"
        reason = "มีคำหยาบปรากฏค่อนข้างบ่อยในหลายช่วงของคลิป"

        detailed_reason = (
            "จัดเป็นเรต 18+ เพราะมีคำหยาบกระจายอยู่ในหลาย segment ของ transcript\n"
            f"- สัดส่วน segment ที่มี profanity = {r_prof:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_18_PROF_RATIO:.2f})\n"
            #"เกณฑ์ที่ใช้: profanity_ratio ≥ THR_18_PROF_RATIO"
        )

    elif max_prof >= THR_18_PROF_MAX:
        rating = "18+"
        reason = "มีคำหยาบที่รุนแรงมากอย่างน้อย 1 ช่วง"

        detailed_reason = (
            "จัดเป็นเรต 18+ เพราะตรวจพบคำหยาบที่มีความรุนแรงสูงจากโมเดล\n"
            f"- ค่าความมั่นใจสูงสุดของ profanity จากโมเดล = {max_prof:.2f} "
            f"(มากกว่าเกณฑ์ {THR_18_PROF_MAX:.2f})\n"
            "ถึงแม้สัดส่วนจะไม่เยอะมาก แต่ความรุนแรงของคำสูงพอให้จัดเป็น 18+\n"
            #"เกณฑ์ที่ใช้: max_profanity_prob ≥ THR_18_PROF_MAX"
        )

    elif r_sex >= THR_18_SEX_RATIO:
        rating = "18+"
        reason = "มีเนื้อหาเชิงเพศในระดับปานกลางกระจายอยู่ในคลิป"

        detailed_reason = (
            "จัดเป็นเรต 18+ เพราะมีเนื้อหาเชิงเพศปรากฏอยู่ในหลายส่วนของ transcript\n"
            f"- สัดส่วน segment ที่เป็น sexual = {r_sex:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_18_SEX_RATIO:.2f})\n"
            f"- ค่าความมั่นใจสูงสุดของ sexual จากโมเดล = {max_sex:.2f}\n"
            #"เกณฑ์ที่ใช้: sexual_ratio ≥ THR_18_SEX_RATIO"
        )

    elif r_hate >= THR_18_HATE_RATIO and max_hate >= THR_18_HATE_MAX:
        rating = "18+"
        reason = "มีเนื้อหา hate speech ต่อกลุ่มคนในระดับที่น่ากังวล"

        detailed_reason = (
            "จัดเป็นเรต 18+ เพราะพบประโยคที่เข้าข่าย hate speech ต่อกลุ่มคนอย่างชัดเจน\n"
            f"- สัดส่วน segment ที่เป็น hate speech = {r_hate:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_18_HATE_RATIO:.2f})\n"
            f"- ค่าความมั่นใจสูงสุดของ hate จากโมเดล = {max_hate:.2f} "
            f"(มากกว่าเกณฑ์ {THR_18_HATE_MAX:.2f})\n"
            #"เกณฑ์ที่ใช้: hate_ratio ≥ THR_18_HATE_RATIO และ max_hate_prob ≥ THR_18_HATE_MAX"
        )

    # ---------- 15+ ----------
    elif r_prof >= THR_15_PROF_RATIO:
        rating = "15+"
        reason = "มีคำหยาบอยู่พอสมควร แต่ยังไม่ถึงระดับ 18+"

        detailed_reason = (
            "จัดเป็นเรต 15+ เพราะมีคำหยาบกระจายอยู่ในคลิประดับหนึ่ง\n"
            f"- สัดส่วน segment ที่มี profanity = {r_prof:.2f} "
            f"(มากกว่าเกณฑ์ขั้นต่ำ {THR_15_PROF_RATIO:.2f})\n"
            f"- แต่ยังไม่ถึงเกณฑ์ 18+ (เช่น {THR_18_PROF_RATIO:.2f})\n"
            #"เกณฑ์ที่ใช้: profanity_ratio อยู่ระหว่าง THR_15_PROF_RATIO และ THR_18_PROF_RATIO"
        )

    elif max_prof >= THR_15_PROF_MAX:
        rating = "15+"
        reason = "มีคำหยาบที่ค่อนข้างแรงอย่างน้อย 1 ช่วง"

        detailed_reason = (
            "จัดเป็นเรต 15+ เพราะพบคำหยาบที่ค่อนข้างแรง แม้จะไม่ได้เกิดขึ้นบ่อยมาก\n"
            f"- ค่าความมั่นใจสูงสุดของ profanity จากโมเดล = {max_prof:.2f} "
            f"(มากกว่าเกณฑ์ {THR_15_PROF_MAX:.2f})\n"
            "แต่ความถี่รวมยังไม่สูงถึงระดับที่ต้องจัดเป็น 18+\n"
            #"เกณฑ์ที่ใช้: max_profanity_prob ≥ THR_15_PROF_MAX"
        )

    # ---------- general ----------
    else:
        rating = "general"
        reason = "เนื้อหาโดยรวมไม่มีคำหยาบ / sexual / ความรุนแรง / hate ในระดับที่ต้องจำกัดมากกว่านี้"

        detailed_reason = (
            "จัดเป็นเรต general เพราะค่าตัวชี้วัดทุกหมวดอยู่ต่ำกว่า threshold ที่กำหนด\n"
            f"- r_prof (profanity ratio) = {r_prof:.2f}\n"
            f"- r_sex  (sexual ratio)    = {r_sex:.2f}\n"
            f"- r_viol (violence ratio)  = {r_viol:.2f}\n"
            f"- r_hate (hate ratio)      = {r_hate:.2f}\n"
            "ทั้งหมดไม่ถึงเกณฑ์ขั้นต่ำของเรต 15+, 18+, หรือ 20+\n"
            #"เกณฑ์ที่ใช้: ทุก ratio < threshold ของแต่ละเรต"
        )

    # ========= debug summary =========
    debug = {
        "n_segments": n_segments,
        "r_prof": r_prof,
        "r_sex": r_sex,
        "r_viol": r_viol,
        "r_hate": r_hate,
        "max_prof": max_prof,
        "max_sex": max_sex,
        "max_viol": max_viol,
        "max_hate": max_hate,
        "has_prof_strong": has_prof_strong,
        "has_sex_strong": has_sex_strong,
        "has_sex_mild": has_sex_mild,
        "has_viol_strong": has_viol_strong,
        "has_viol_mild": has_viol_mild,
        "has_hate_slur": has_hate_slur,
    }

    return rating, reason, detailed_reason, debug
