import re
import argparse
from datetime import datetime, timedelta

import pandas as pd

# =========================
# 1) Advanced clean_text
# =========================

LEET_MAP = str.maketrans({
    "@": "a",
    "$": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "3": "e",
    "4": "a",
    "7": "t",
})

PROFANITY_PATTERNS = [
    # fuck
    (r"f[\W_]*u[\W_]*c[\W_]*k+", "fuck"),
    (r"f+[\W_]*\*+[\W_]*k+", "fuck"),
    # shit
    (r"s[\W_]*h[\W_]*i[\W_]*t+", "shit"),
    # bitch
    (r"b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+", "bitch"),
    # ass / asshole
    (r"a[\W_]*s[\W_]*s+", "ass"),
    (r"a[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e+", "asshole"),
    # damn
    (r"d[\W_]*a[\W_]*m[\W_]*n+", "damn"),
    # porn
    (r"p[\W_]*o[\W_]*r[\W_]*n+", "porn"),
]

URL_RE = re.compile(r"http\S+|www\.\S+")

def clean_text_advanced(s: str) -> str:
    """normalize + decensor ข้อความ (ใช้ตอนสร้าง CSV infer)"""
    s = str(s).strip().lower()
    # ตัด URL
    s = URL_RE.sub(" ", s)
    # แปลง leet
    s = s.translate(LEET_MAP)
    # ตัด emoji/unicode สูง ๆ
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)
    # ลบ *, ., _ กลางคำ เช่น f***, b.i.t.c.h
    s = re.sub(r"(?<=\w)[\*\._]+(?=\w)", "", s)

    for pat, repl in PROFANITY_PATTERNS:
        s = re.sub(pat, repl, s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# 2) Parsing เวลา
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
# 3) แปลง TXT -> CSV
# =========================

def txt_to_csv(input_path: str, output_path: str):
    # เตรียม regex รองรับหลายแพทเทิร์น
    # 1) 00:00:00 - 00:00:05    text
    range_pat = re.compile(
        r"^\s*(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s+(.+)$"
    )
    # 2) [00:00:00] text
    bracket_pat = re.compile(
        r"^\s*\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$"
    )
    # 3) 00:00:12.760 text
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
                rows.append({
                    "start_time": start,
                    "end_time": end,
                    "text": text.strip()
                })
            elif m_br:
                start_raw, text = m_br.groups()
                start = norm_time_str(start_raw)
                rows.append({
                    "start_time": start,
                    "end_time": None,
                    "text": text.strip()
                })
            elif m_plain:
                start_raw, text = m_plain.groups()
                start = norm_time_str(start_raw)
                rows.append({
                    "start_time": start,
                    "end_time": None,
                    "text": text.strip()
                })
            else:
                # บรรทัดที่แพทเทิร์นไม่เข้า (จะข้ามไป หรือจะ log ก็ได้)
                # print("Skip line:", line)
                continue

    # เติม end_time ถ้าไม่มี (ใช้ start ของบรรทัดถัดไป / +3s สำหรับบรรทัดสุดท้าย)
    for i, row in enumerate(rows):
        if row["end_time"] is None:
            if i < len(rows) - 1:
                row["end_time"] = rows[i + 1]["start_time"]
            else:
                row["end_time"] = add_seconds_str(row["start_time"], 3)

    # ทำ clean_text advanced
    for row in rows:
        row["text"] = clean_text_advanced(row["text"])

    df = pd.DataFrame(rows, columns=["start_time", "end_time", "text"])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved CSV to: {output_path}")
    print(df.head())

# =========================
# 4) main (ใช้ argparse)
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert transcript TXT → CSV (start_time,end_time,text)."
    )
    parser.add_argument(
        "--txt",
        type=str,
        required=False,
        default="test.txt",
        help="พาธไฟล์ .txt ต้นฉบับ (default: test.txt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=False,
        default="outputs/transcripts/transcript_clean.csv",
        help="พาธไฟล์ .csv ปลายทาง (default: outputs/transcripts/transcript_clean.csv)",
    )
    args = parser.parse_args()

    txt_to_csv(args.txt, args.out)

# normalize → เก็บเฉพาะ a-z, ตัวเลข, เว้นวรรค และ . ! ? ' (ไว้ตัดประโยค)
'''text = text.lower()
text = re.sub(r"[^a-z0-9\s\.\!\?\']", " ", text)
text = re.sub(r"\s+", " ", text).strip()

# de-censor (ขยายเพิ่มได้)
patterns = {
    r"f[\*\-_\.\s]?u[\*\-_\.\s]?c[\*\-_\.\s]?k": "fuck",
    r"s[\*\-_\.\s]?h[\*\-_\.\s]?i[\*\-_\.\s]?t": "shit",
    r"b[\*\-_\.\s]?i[\*\-_\.\s]?t[\*\-_\.\s]?c[\*\-_\.\s]?h": "bitch",
    r"d[\*\-_\.\s]?i[\*\-_\.\s]?c[\*\-_\.\s]?k": "dick",
    r"p[\*\-_\.\s]?u[\*\-_\.\s]?s[\*\-_\.\s]?s[\*\-_\.\s]?y": "pussy",
    r"c[\*\-_\.\s]?u[\*\-_\.\s]?n[\*\-_\.\s]?t": "cunt"
}
for pat, rep in patterns.items():
    text = re.sub(pat, rep, text)

# ตัดเป็นประโยค
sentences = [s.strip() for s in re.split(r"[\.!\?]+\s+", text) if len(s.strip()) >= 5]
pd.DataFrame({"sentence": sentences}).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"saved {OUT} with {len(sentences)} sentences")'''
