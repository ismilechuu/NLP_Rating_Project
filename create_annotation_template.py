#!/usr/bin/env python3
# create_annotation_template.py
# Create annotation template by selecting 200 segments from existing transcripts

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_transcript_segments(file_path):
    """Load segments from a transcript file"""
    # Check if it's a CSV or TXT file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if 'text' in df.columns:
            return df[['text']].copy()
    elif file_path.endswith('.txt'):
        # Parse TXT file (format: [timestamp] text)
        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    # Format: [timestamp]\ttext
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        segments.append({'text': parts[1]})
        return pd.DataFrame(segments)
    
    return pd.DataFrame()


def stratified_sample(transcripts_dict, n_per_file=50):
    """
    Stratified sampling: take n_per_file segments from each transcript
    """
    all_samples = []
    
    for file_name, df in transcripts_dict.items():
        if len(df) == 0:
            continue
        
        # Sample n_per_file segments (or all if less than n_per_file)
        n_sample = min(n_per_file, len(df))
        
        # Random sampling
        sampled = df.sample(n=n_sample, random_state=42)
        sampled['source_file'] = file_name
        
        all_samples.append(sampled)
        print(f"✅ Sampled {n_sample} segments from {file_name}")
    
    # Combine all samples
    combined = pd.concat(all_samples, ignore_index=True)
    
    return combined


def create_annotation_template(output_path, segments_df):
    """
    Create annotation template CSV with empty label columns
    """
    # Add empty label columns
    segments_df['profanity'] = ''
    segments_df['sexual'] = ''
    segments_df['violence'] = ''
    segments_df['hate'] = ''
    segments_df['notes'] = ''
    
    # Reorder columns
    columns = ['text', 'profanity', 'sexual', 'violence', 'hate', 'notes', 'source_file']
    segments_df = segments_df[columns]
    
    # Save to CSV
    segments_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return segments_df


def main():
    parser = argparse.ArgumentParser(
        description="Create annotation template from existing transcripts"
    )
    parser.add_argument(
        "--out",
        default="annotation_template.csv",
        help="Output CSV file for annotation (default: annotation_template.csv)"
    )
    parser.add_argument(
        "--n_per_file",
        type=int,
        default=50,
        help="Number of segments to sample per file (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📝 Creating Annotation Template")
    print("=" * 80)
    print()
    
    # List of transcript files to sample from
    transcript_files = [
        'transcript_4.txt',
        '15+_profane_violent_hate_sexual.txt',
        '18+_hate_sexual_violent_profane.txt',
        '20+_hate_sexual_violent_profane.txt',
        'General_gerneral_profane.txt'
    ]
    
    # Also check for CSV files in outputs
    csv_files = [
        'outputs/transcript_4_with_scores.csv',
        'outputs/15+_profane_violent_hate_sexual_with_scores.csv',
        'outputs/18+_hate_sexual_violent_profane_with_scores.csv',
        'outputs/20+_hate_sexual_violent_profane_with_scores.csv',
        'outputs/General_gerneral_profane_with_scores.csv'
    ]
    
    # Load segments from each file
    transcripts = {}
    
    print("📂 Loading transcript files...")
    print()
    
    # Try TXT files first
    for file_path in transcript_files:
        if Path(file_path).exists():
            df = load_transcript_segments(file_path)
            if len(df) > 0:
                transcripts[file_path] = df
                print(f"   ✅ {file_path}: {len(df)} segments")
    
    # Try CSV files
    for file_path in csv_files:
        if Path(file_path).exists():
            file_name = Path(file_path).stem.replace('_with_scores', '')
            if file_name not in [Path(f).stem for f in transcripts.keys()]:
                df = load_transcript_segments(file_path)
                if len(df) > 0:
                    transcripts[file_path] = df
                    print(f"   ✅ {file_path}: {len(df)} segments")
    
    if not transcripts:
        print("❌ No transcript files found!")
        return
    
    print()
    print(f"📊 Total files loaded: {len(transcripts)}")
    print()
    
    # Stratified sampling
    print(f"🎲 Sampling {args.n_per_file} segments from each file...")
    print()
    sampled_segments = stratified_sample(transcripts, n_per_file=args.n_per_file)
    
    print()
    print(f"📊 Total segments sampled: {len(sampled_segments)}")
    print()
    
    # Create annotation template
    print(f"📝 Creating annotation template: {args.out}")
    template_df = create_annotation_template(args.out, sampled_segments)
    
    print()
    print("=" * 80)
    print("✅ Annotation template created successfully!")
    print("=" * 80)
    print()
    print("📋 Instructions for annotation:")
    print()
    print("1. Open the CSV file in Excel or Google Sheets")
    print("2. For each row, fill in the label columns:")
    print("   - profanity: 0 (no) or 1 (yes)")
    print("   - sexual:    0 (no) or 1 (yes)")
    print("   - violence:  0 (no) or 1 (yes)")
    print("   - hate:      0 (no) or 1 (yes)")
    print("3. Use 'notes' column for any comments or ambiguous cases")
    print("4. Save the file when done")
    print()
    print(f"💾 File saved: {args.out}")
    print()


if __name__ == "__main__":
    main()

