"""
Generates a pooled 80/20 train-test split and saves it to splits/pooled_80_20.json
This is the split I will use throughout the analysis and running models

Default: 
- 20% of each dataset is in test, rest in train
- saves to splits/pooled_80_20.json

Commands to copy:
  uv run python scripts/make_pooled_split.py
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_epoch_table
from src.split import make_pooled_split


def main():
    parser = argparse.ArgumentParser(description="Generate pooled train/test split and save")
    parser.add_argument("--out", type=str, default="splits/pooled_80_20")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subj_df = get_epoch_table().drop_duplicates("subject_id")[["subject_id", "dataset"]]
    make_pooled_split(subj_df, args.out, test_frac=args.test_frac, random_state=args.seed)


if __name__ == "__main__":
    main()
