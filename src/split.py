"""
Make the train/test splits to feed into the run_all_models.py file

Split by subject: a subject's entire rest pattern (all 30 epochs) is
either fully in train or fully in test. Do not mix epochs from the same
subject across train and test.

Can split by dataset (train = original, test = replication) or 
split by test_frac (certain percentage of each dataset is train, rest is test)

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def split_by_dataset(
    df,
    train_dataset="original",
    val_dataset="replication",
):
    """
    df = all data

    Train on original dataset, test on replication.
    All epochs from all subjects in train_dataset go to train; all epochs from
    all subjects in val_dataset go to test.
    """
    train = df[df["dataset"] == train_dataset].copy()
    val = df[df["dataset"] == val_dataset].copy()

    return train, val

# split by test fraction
# default is 80/20 (so 80% of subjects of each dataset goes to train, rest go to test)
def train_test_split_pooled(df, test_frac=0.2, random_state=42):
    """
    Take test_frac of subjects from each dataset (original, replication) as test; the rest as train.
    This is the "pooled" option.
    """
    rng = np.random.default_rng(random_state)
    train_subjects, test_subjects = [], []
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        subjects = np.array(subset["subject_id"].unique())
        rng.shuffle(subjects)
        n_test = max(1, int(len(subjects) * test_frac))
        test_ids = set(subjects[:n_test])
        train_ids = set(subjects[n_test:])
        train_subjects.append(subset[subset["subject_id"].isin(train_ids)])
        test_subjects.append(subset[subset["subject_id"].isin(test_ids)])
    train = pd.concat(train_subjects, ignore_index=True)
    test = pd.concat(test_subjects, ignore_index=True)
    return train, test


def save_split_ids(
    train_ids,
    test_ids,
    path,
):
    """Save train and test subject IDs to a JSON file to reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    with open(path, "w") as f:
        json.dump({"train": sorted(train_ids), "test": sorted(test_ids)}, f, indent=2)
    print(f"Saved split IDs to {path}")


def make_pooled_split(subj_df, path, test_frac=0.2, random_state=42):
    """
    Create a pooled train/test split by subject (stratified by dataset)
    save to JSON at path
    subj_df must have columns subject_id and dataset (one row per subject).
    """
    train_df, test_df = train_test_split_pooled(subj_df, test_frac=test_frac, random_state=random_state)
    save_split_ids(set(train_df["subject_id"]), set(test_df["subject_id"]), path)

def load_split_ids(path):
    """Load train and test subject IDs from a JSON file 
    so we can reuse the same sets that we saved."""
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    with open(path) as f:
        data = json.load(f)
    return set(data["train"]), set(data["test"])


def split_df_by_subject_ids(df, train_ids, test_ids):
    """Split dataframe into train and test by subject_id."""
    train = df[df["subject_id"].isin(train_ids)].copy()
    test = df[df["subject_id"].isin(test_ids)].copy()
    return train, test


def kfold_by_subject(df, n_splits=15, random_state=42):
    """
    K-fold cross-validation by subject.
    Each subject's epochs are entirely in train or entirely in val for a given fold.
    Returns (train_df, val_df) per fold
    """
    groups = df["subject_id"].values
    n_subjects = df["subject_id"].nunique()

    if n_splits > n_subjects:
        raise ValueError(f"n_splits ({n_splits}) can't be larger than ({n_subjects})")

    try:
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except TypeError:
        cv = GroupKFold(n_splits=n_splits)
    
    folds = []
    for train_idx, val_idx in cv.split(df, groups=groups):
        folds.append((df.iloc[train_idx].copy(), df.iloc[val_idx].copy()))

    return folds