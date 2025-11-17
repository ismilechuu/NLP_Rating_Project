import re
import argparse
from datetime import datetime, timedelta
import os

import pandas as pd

# =========================
# 1) Profanity / URL helpers
# =========================

# แปลง leet เฉพาะตัวอักษร ไม่แตะตัวเลข (กัน 20 -> 2o)
'''LEET_MAP = str.maketrans({
    "@": "a",
    "$": "s",
    "!": "i",
    "3": "e",
    "4": "a",
    "7": "t",
})'''

PROFANITY_PATTERNS = [
    # fuck
    (r"f[\W_]*u[\W_]*c[\W_]*k+", "fuck"),
    (r"f+[\W_]*\*+[\W_]*k+", "fuck"),
    # shit
    (r"s[\W_]*h[\W_]*i[\W_]*t+", "shit"),
    # bitch
    (r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+", "bitch"),
    # ass / asshole
    (r"a[\W_]*s[\W_]*s[\W_]*h?[\W_]*o?[\W_]*l?[\W_]*e?", "ass"),
    # damn
    (r"d[\W_]*a[\W_]*m[\W_]*n+", "damn"),
    # porn
    (r"p[\W_]*o[\W_]*r[\W_]*n+", "porn"),
]

URL_RE = re.compile(r"http\S+|www\.\S+")


# =========================
# 2) ฟังก์ชัน clean ข้อความ
# =========================

def clean_text(s: str) -> str:
    if pd.isna(s):
        return s

    s = str(s)

    # 1) normalize smart quotes → ปกติ
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")

    # 2) strip space
    s = s.strip()

    # 3) ตัด quote ซ้อน "" "" '' ''
    while (
        len(s) >= 2
        and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'"))
    ):
        s = s[1:-1].strip()

    # 4) lowercase
    s = s.lower()

    # 5) remove URL
    s = URL_RE.sub(" ", s)

    # 6) leet transform (only letters)
    '''s = s.translate(LEET_MAP)'''

    # 7) remove emoji / unicode
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)

    # 8) remove censor patterns f***, b.i.t.c.h
    s = re.sub(r"(?<=\w)[\*\._]+(?=\w)", "", s)

    # 9) ตัด possessive 's เช่น night's → night
    s = re.sub(r"\b(\w+)'s\b", r"\1", s)

    # 10) decensor profanity
    for pat, repl in PROFANITY_PATTERNS:
        s = re.sub(pat, repl, s)

    # 11) remove ellipsis: … or ... 
    s = re.sub(r"[.…]+", " ", s)

    # 12) normalize duplicate punctuation: !! → !, ?? → ?
    s = re.sub(r"(!){2,}", "!", s)
    s = re.sub(r"(\?){2,}", "?", s)

    # 13) normalize space
    s = re.sub(r"\s+", " ", s).strip()

    return s



# =========================
# 3) ฟังก์ชันช่วยจัดการเวลา (ตอนอ่านจาก TXT)
# =========================

def norm_time_str(t: str) -> str:
    """รับ 'HH:MM:SS' หรือ 'HH:MM:SS.xxx' -> คืน 'HH:MM:SS'"""
    t = t.strip()
    fmt = "%H:%M:%S.%f" if "." in t else "%H:%M:%S"
    dt = datetime.strptime(t, fmt)
    return dt.strftime("%H:%M:%S")


def add_seconds_str(t: str, sec: int) -> str:
    """เพิ่มวินาทีให้เวลา (string)"""
    t = t.strip()
    fmt = "%H:%M:%S.%f" if "." in t else "%H:%M:%S"
    dt = datetime.strptime(t, fmt)
    dt2 = dt + timedelta(seconds=sec)
    return dt2.strftime("%H:%M:%S")


# =========================
# 4) อ่าน TXT → rows + clean text
# =========================

def parse_txt_to_rows(input_path: str):
    """
    รองรับรูปแบบเวลา:
    - 00:00:00 - 00:00:05 text
    - [00:00:00] text
    - 00:00:00 text
    แล้วคืน list ของ dict {start_time, end_time, text}
    """

    # 00:00:00 - 00:00:05 text  (รองรับ '-', '–', '—')
    range_pat = re.compile(
        r"^\s*"
        r"(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)"
        r"\s*[-–—]\s*"
        r"(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)"
        r"\s+(.+)$"
    )

    # [00:00:00] text
    bracket_pat = re.compile(
        r"^\s*\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$"
    )

    # 00:00:00 text
    plain_pat = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s+(.+)$"
    )

    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            m_range = range_pat.match(line)
            m_br = bracket_pat.match(line)
            m_plain = plain_pat.match(line)

            if m_range:
                start_raw, end_raw, text = m_range.groups()
                start = norm_time_str(start_raw)
                end = norm_time_str(end_raw)

                # กันเคส end <= start หรือ 00:00:00
                if end <= start or end == "00:00:00":
                    end = None

                rows.append({
                    "start_time": start,
                    "end_time": end,
                    "text": clean_text(text),
                })

            elif m_br:
                start_raw, text = m_br.groups()
                start = norm_time_str(start_raw)

                rows.append({
                    "start_time": start,
                    "end_time": None,
                    "text": clean_text(text),
                })

            elif m_plain:
                start_raw, text = m_plain.groups()
                start = norm_time_str(start_raw)

                rows.append({
                    "start_time": start,
                    "end_time": None,
                    "text": clean_text(text),
                })
            else:
                # format เพี้ยนมาก ข้ามบรรทัดนั้นไป
                # ถ้าอยาก debug เปิด print ได้
                # print("Skip line:", line)
                continue

    # เติม end_time ที่ว่างด้วย start ของบรรทัดถัดไป หรือ +3 วิ
    for i, row in enumerate(rows):
        if row["end_time"] is None:
            if i < len(rows) - 1:
                row["end_time"] = rows[i + 1]["start_time"]
            else:
                row["end_time"] = add_seconds_str(row["start_time"], 3)

    return rows


# =========================
# 5) main logic: รับได้ทั้ง TXT และ CSV
# =========================

def process_file(input_path: str, output_path: str):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".txt":
        # 1) แปลง TXT → rows + clean text
        rows = parse_txt_to_rows(input_path)
        if not rows:
            print("⚠️ ไม่พบบรรทัดที่เป็น transcript ในไฟล์:", input_path)
            df = pd.DataFrame(columns=["start_time", "end_time", "text"])
        else:
            df = pd.DataFrame(rows, columns=["start_time", "end_time", "text"])
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ TXT → CSV พร้อม clean text แล้ว: {output_path}")

    else:
        # assume เป็น CSV: อ่านแล้ว clean เฉพาะ column text
        df = pd.read_csv(input_path)
        if "text" not in df.columns:
            print("⚠️ ไม่พบคอลัมน์ 'text' ใน CSV จะไม่แก้อะไร")
        else:
            df["text"] = df["text"].apply(clean_text)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Clean text ใน CSV เสร็จแล้ว: {output_path}")


# =========================
# 6) CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="รับไฟล์ .txt หรือ .csv แล้วแปลง/clean text ให้พร้อมใช้กับโมเดล"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="พาธไฟล์ .txt หรือ .csv ต้นฉบับ",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        required=False,
        default="outputs/transcripts/transcript_clean.csv",
        help="พาธไฟล์ .csv ปลายทาง (default: outputs/transcripts/transcript_clean.csv)",
    )
    args = parser.parse_args()

    process_file(args.input, args.out)
