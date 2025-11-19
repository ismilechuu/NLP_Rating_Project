#!/usr/bin/env python3
"""
แปลง label_complete.csv เป็น ground_truth.csv
ที่มี format ตรงกับที่ระบบต้องการ

Usage:
    python prepare_ground_truth.py
"""

import pandas as pd
import argparse


def prepare_ground_truth(input_csv='label_complete.csv', output_csv='ground_truth.csv'):
    """
    แปลง label_complete.csv เป็น ground_truth.csv
    
    Input format:
        id,time_str,sentence,,,ground_truth_profane,ground_truth_violent,ground_truth_sexual,ground_truth_hate
    
    Output format:
        text,profanity,sexual,violence,hate
    """
    print(f"📥 อ่านไฟล์: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"✅ โหลดข้อมูล: {len(df)} samples")
    
    # เลือกและเปลี่ยนชื่อ columns
    ground_truth = df[['sentence', 'ground_truth_profane', 'ground_truth_violent', 
                        'ground_truth_sexual', 'ground_truth_hate']].copy()
    
    ground_truth.columns = ['text', 'profanity', 'violence', 'sexual', 'hate']
    
    # เรียงลำดับ columns ให้ตรงกับระบบ (text, profanity, sexual, violence, hate)
    ground_truth = ground_truth[['text', 'profanity', 'sexual', 'violence', 'hate']]
    
    # แสดง label distribution
    print("\n📊 Label Distribution:")
    print(f"  Profanity: {ground_truth['profanity'].sum()} samples ({ground_truth['profanity'].sum()/len(ground_truth)*100:.1f}%)")
    print(f"  Sexual:    {ground_truth['sexual'].sum()} samples ({ground_truth['sexual'].sum()/len(ground_truth)*100:.1f}%)")
    print(f"  Violence:  {ground_truth['violence'].sum()} samples ({ground_truth['violence'].sum()/len(ground_truth)*100:.1f}%)")
    print(f"  Hate:      {ground_truth['hate'].sum()} samples ({ground_truth['hate'].sum()/len(ground_truth)*100:.1f}%)")
    
    # บันทึก
    ground_truth.to_csv(output_csv, index=False)
    print(f"\n✅ บันทึกไฟล์: {output_csv}")
    print(f"   จำนวน: {len(ground_truth)} samples")
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="แปลง label_complete.csv เป็น ground_truth.csv")
    parser.add_argument('--input', default='label_complete.csv', help='Input CSV file')
    parser.add_argument('--output', default='ground_truth.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    prepare_ground_truth(args.input, args.output)


if __name__ == "__main__":
    main()

