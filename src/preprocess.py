"""
Load blockwise epoch-level data and build features for the baselines + lstm.

baselines need epoch-level features, lstm needs trial-level features

Data: 
- cleaned_exp_data/original_blockwise_cleaned.csv and 
cleaned_exp_data/replication_blockwise_cleaned.csv
- replication_main_trials_cleaned.csv and original_main_trials_cleaned.csv
* the epoch level files are derived beforehand from the main trial files

original subject ids are original_{subj_id}
replication subject ids are replication_{subj_id}
"""
from pathlib import Path
import pandas as pd
import numpy as np

# leads to data folder
_DATA_DIR = Path(__file__).resolve().parents[1] / "cleaned_exp_data"

# Trial types that are main task responses (exclude rt_main_trials for trial/epoch counts).
MAIN_RESPONSE_TRIAL_TYPES = ("ds_main_response", "sr_main_response")

################################################################################
# LOADING TRIAL LEVEL FILES: 
# original is from the 2024 FYP experiment
def load_original_trials():
    """Load original experiment trials; Adds dataset and edits subject_id."""
    path = _DATA_DIR / "original_main_trials_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "original"
    df["subject_id"] = "original_" + df["subj_id"].astype(str)
    return df

# this is from the 2025 replication experiment (cf_ts_rep)
def load_replication_trials():
    """Load replication experiment trials; Adds dataset and edits subject_id"""
    path = _DATA_DIR / "replication_main_trials_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "replication"
    df["subject_id"] = "replication_" + df["subj_id"].astype(str)
    return df

################################################################################
# LOADING EPOCH LEVEL FILES: 
# load the epoch-level files:
# original is from the 2024 FYP experiment
def load_original():
    """Load original experiment blockwise (epoch-level) data. Adds dataset and edits subject_id"""
    path = _DATA_DIR / "original_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "original"
    df["subject_id"] = "original_" + df["subj_id"].astype(str)
    return df

# this is from the 2025 replication experiment (cf_ts_rep)
def load_replication():
    """Load replication experiment blockwise (epoch-level) data. Adds dataset and edits subject_id"""
    path = _DATA_DIR / "replication_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "replication"
    df["subject_id"] = "replication_" + df["subj_id"].astype(str)
    return df

################################################################################
# PROCESSING EPOCH LEVEL FILES: (used for the baselines)

def get_num_timeouts_per_epoch():
    """Number of timed-out trials per (subject_id, epoch_num) (only for main games)"""
    original = load_original_trials()
    original = assign_epoch_num_original(original)
    replication = load_replication_trials()
    both = pd.concat([original, replication], ignore_index=True)
    main_trials_only = both[both["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)]

    if ("timed_out" not in main_trials_only.columns):
        print("issue: main_trials do not have timed_out column")

    # returns one row per subj/epoch with number of timed-out trials
    return (
        main_trials_only
        .groupby(["subject_id", "epoch_num"])["timed_out"]
        .apply(lambda x: x.astype(bool).sum())
        .reset_index(name="num_timeouts")
    )

def get_epoch_table():
    """
    Processes an updated blockwise file from both datasets 
    (including trial data that we calculate)
    renames rest_length from num_rest_in_chunk, includes all other included blockwise variables,
    and num_timeouts (calculated from trial-level files in get_num_timeouts_per_epoch()).
    """
    original = load_original()
    replication = load_replication()
    df = pd.concat([original, replication], ignore_index=True)
    df = df.rename(columns={"num_rest_in_chunk": "rest_length"})
    df = df.sort_values(["dataset", "subj_id", "epoch_num"]).reset_index(drop=True)
    timeouts_per_epoch = get_num_timeouts_per_epoch()
    
    df = df.merge(timeouts_per_epoch, on=["subject_id", "epoch_num"], how="left")
    
    # check just in case something is wrong and there are epoch with no time out data
    if df["num_timeouts"].isna().any():
        print("issue: some epochs have no timeout data, filling with 0 for now but go check")
        df["num_timeouts"] = df["num_timeouts"].fillna(0)

    return df

################################################################################
# PROCESSING TRIAL LEVEL FILES: (used for the LSTM)

# original dataset specific edits:
def assign_epoch_num_original(trials):
    """
    In original dataset each epoch was actually called a block, and each block was called a group.
    Here we re-name the blocks to epochs and the groups to blocks. epochs should be 1 to 30.
    blocks should be 1 to 10. each epoch is one continuous block of main trials.
    """
    out = trials.copy().sort_values(["subject_id", "trial_index"]).reset_index(drop=True)
    out = out.rename(columns={"block_num": "epoch_num", "group_num": "block_num"})

    return out

# get all the trial data from both datasets:
def get_trials():
    """
    Combined trial-level table from both experiments.
    Both datasets have subject_id, dataset, rt, game_type, is_correct_numeric, block_num, num_rest_in_chunk, etc.
    merges cue_transition_type (stay_within_block,
    stay_between_block, switch_between_block) from the blockwise
    data with each trial by (subject_id, epoch_num).
    """
    original = load_original_trials()
    replication = load_replication_trials()
    original = assign_epoch_num_original(original)

    # make sure we're only keeping main trials
    original = original[original["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)]
    replication = replication[replication["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)]

    if "epoch_num" not in replication.columns:
        print("issue: replication does not have epoch_num column")
    if "epoch_num" not in original.columns:
        print("issue: original does not have epoch_num column")

    # Merge blockwise epoch-level cols into trials
    epoch_table = get_epoch_table()
    req_cols = ["subject_id", "epoch_num", "cue_transition_type", "rest_type"]
    epoch_reqs = epoch_table[req_cols].drop_duplicates()
    original = original.merge(epoch_reqs, on=["subject_id", "epoch_num"], how="left")
    replication = replication.merge(epoch_reqs, on=["subject_id", "epoch_num"], how="left")
    return pd.concat([original, replication], ignore_index=True)


################################################################################
# ADDING FEATURES TO THE EPOCH LEVEL FILES: (used for the baselines)

# add baseline features to the epoch level file:
def add_baseline_features(df):
    """
    takes in an epoch level table df and adds: (this is for the two baselines)
    - cue_transition_type (3-level dummies)
    - game_type
    - block_num
    - accuracy_sd
    - rt_sd
    - num_timeouts
    - rest_length
    - rest_length_prev
    - accuracy_prev
    """
    all_features_df = df.copy()

    # cue transition type
    # reference = stay_within_block
    cue_transition_type = all_features_df["cue_transition_type"].astype(str)
    all_features_df["cue_stay_between_block"] = (cue_transition_type == "stay_between_block").astype(int)
    all_features_df["cue_switch_between_block"] = (cue_transition_type == "switch_between_block").astype(int)

    # game type (default reference = spatial_recall)
    game_type = all_features_df["game_type"].astype(str)
    all_features_df["game_type_digit_span"] = (game_type == "digit_span").astype(int)

    # block num
    all_features_df["block_num"] = all_features_df["block_num"].astype(int)

    # accuracy
    all_features_df["accuracy_sd"] = all_features_df["accuracy_sd"].fillna(0)

    # rt
    all_features_df["rt_sd"] = all_features_df["rt_sd"].fillna(0)
    if (all_features_df["avg_rt"].isna().any()):
        print("some epochs have no rt data (means all were skipped), filling with max possible rt")
        all_features_df["avg_rt"] = all_features_df["avg_rt"].fillna(all_features_df["avg_rt"].max())

    # timeouts
    all_features_df["num_timeouts"] = all_features_df["num_timeouts"].astype(int)

    # rest length
    all_features_df["rests_taken_so_far"] = (
        all_features_df.groupby("subject_id")["rest_length"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
        .astype(float)
    )

    # dataset (default reference = original)
    all_features_df["dataset_replication"] = (all_features_df["dataset"] == "replication").astype(int)

    # add cumulative accuracy by game type SO FAR (so excludes current epoch)
    # game_type already exists from above
    for game, col_name in [("digit_span", "avg_accuracy_until_now_digit_span"),
                            ("spatial_recall", "avg_accuracy_until_now_spatial_recall")]:
        acc = all_features_df["avg_epoch_accuracy"].where(game_type == game)
        all_features_df[col_name] = (
            acc.groupby(all_features_df["subject_id"])
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        ).fillna(0.5)

    return all_features_df

# additional features for history baselines:
def add_history_baseline_features(df):
    """previous-epoch features within each subject: 
    rest_length_prev, accuracy_prev, rt_prev, digit_span_prev, 
    cue_stay_between_block_prev, cue_switch_between_block_prev"""

    history_df = df.copy()
    history_df = df.copy().sort_values(["subject_id", "epoch_num"]).reset_index(drop=True)

    grouped_df = history_df.groupby("subject_id")
    history_df["rest_length_prev"] = grouped_df["rest_length"].shift(1)
    history_df["accuracy_prev"] = grouped_df["avg_epoch_accuracy"].shift(1)
    history_df["rt_prev"] = grouped_df["avg_rt"].shift(1)

    history_df["game_type_digit_span_prev"] = grouped_df["game_type_digit_span"].shift(1)
    history_df["previous_cue_stay_between_block"] = grouped_df["cue_stay_between_block"].shift(1)
    history_df["previous_cue_switch_between_block"] = grouped_df["cue_switch_between_block"].shift(1)

    return history_df

# get the column name for the features of the relevant baseline:
def get_feature_columns(baseline_only=False, include_dataset=False):
    """column names for the model features. 
    if baseline_only = False then we do add the history features
    if include_dataset = True then we add the dataset id (original vs replication)
    ^ the dataset id is only for the pooled split"""

    baseline = [
        "epoch_num",
        "avg_epoch_accuracy",
        "accuracy_sd",
        "avg_rt",
        "rt_sd",
        "num_timeouts",
        "game_type_digit_span",
        "cue_stay_between_block",
        "cue_switch_between_block",
        "rests_taken_so_far",
        "avg_accuracy_until_now_digit_span",
        "avg_accuracy_until_now_spatial_recall",
    ]

    # need dataset and no history
    if include_dataset and baseline_only:
        baseline.append("dataset_replication")
        return baseline
    # need no dataset and no history
    if baseline_only:
        return baseline
    # need dataset and history
    if include_dataset:
        return baseline + [
            "dataset_replication",
            "rest_length_prev",
            "accuracy_prev",
            "rt_prev",
            "game_type_digit_span_prev",
            "previous_cue_stay_between_block",
            "previous_cue_switch_between_block",
        ]
    # need no dataset and yes history
    return baseline + [
            "rest_length_prev",
            "accuracy_prev",
            "rt_prev",
            "game_type_digit_span_prev",
            "previous_cue_stay_between_block",
            "previous_cue_switch_between_block",
        ]

################################################################################
# ADDING FEATURES TO THE TRIAL LEVEL FILES: (used for the LSTM)

def add_trial_features(trials, epoch_table=None):
    """
    Add trial-level features for LSTM: game_type dummies, cue dummies, timed_out,
    is_correct_numeric, rt, epoch_num, block_num, rest_used_so_far.
    """
    all_features_df = trials.copy()
    # Game type
    game_type = all_features_df["game_type"].astype(str)
    all_features_df["game_type_digit_span"] = (game_type == "digit_span").astype(int)

    # Cue dummies (from blockwise merge)
    cue_transition_type = all_features_df["cue_transition_type"].astype(str)
    all_features_df["cue_stay_between_block"] = (cue_transition_type == "stay_between_block").astype(int)
    all_features_df["cue_switch_between_block"] = (cue_transition_type == "switch_between_block").astype(int)

    # Timed out
    if "timed_out" in all_features_df.columns:
        to = all_features_df["timed_out"].astype(str).str.strip().str.lower()
        all_features_df["timed_out_flag"] = to.isin(("true", "1", "1.0")).astype(int)
    else:
        print("issue: LSTM preprocessing: trials do not have timed_out column")
        all_features_df["timed_out_flag"] = 0
    
    # is_correct_numeric
    all_features_df["is_correct"] = all_features_df["is_correct_numeric"].fillna(0).astype(float)

    # rt: fill NA(timeouts) with max possible rt
    if all_features_df["rt"].isna().any():
        all_features_df["rt"] = all_features_df["rt"].fillna(all_features_df["rt"].max())
    all_features_df["rt"] = all_features_df["rt"].astype(float)

    # epoch_num and block_num
    all_features_df["epoch_num"] = all_features_df["epoch_num"].astype(int)
    all_features_df["block_num"] = all_features_df["block_num"].astype(int)

    # rest_used_so_far: cumulative rest from previous completed epochs
    if epoch_table is None:
        epoch_table = get_epoch_table()

    rest_per_epoch = epoch_table[["subject_id", "epoch_num", "rest_length"]].drop_duplicates()
    rest_per_epoch = rest_per_epoch.sort_values(["subject_id", "epoch_num"]).reset_index(drop=True)
    cumulative_rest = rest_per_epoch.groupby("subject_id")["rest_length"].cumsum()
    rest_per_epoch["cumulative_rest_before"] = cumulative_rest - rest_per_epoch["rest_length"]

    all_features_df = all_features_df.merge(
        rest_per_epoch[["subject_id", "epoch_num", "cumulative_rest_before"]],
        on=["subject_id", "epoch_num"],
        how="left",
    )
    all_features_df["rest_used_so_far"] = all_features_df["cumulative_rest_before"].fillna(0).astype(float)
    all_features_df = all_features_df.drop(columns=["cumulative_rest_before"])

    # dataset_replication
    all_features_df["dataset_replication"] = (all_features_df["dataset"] == "replication").astype(int)

    return all_features_df

# get column names for trial level feature (for lstm)
def get_trial_feature_columns(include_dataset=False):
    """trial-level features (col names)"""
    cols = [
        "epoch_num",
        "is_correct",
        "rt",
        "timed_out_flag",
        "game_type_digit_span",
        "cue_stay_between_block",
        "cue_switch_between_block",
        "rest_used_so_far",
    ]
    if include_dataset:
        cols.append("dataset_replication")
    return cols

################################################################################
# GETTING THE DATA: (calls the functions above)

# for the baselines + history baselines:
def get_data():
    """Epoch table, baseline features,history features"""
    all_data = get_epoch_table()
    all_data = add_baseline_features(all_data)
    all_data = add_history_baseline_features(all_data)
    return all_data

# for the LSTM:
def build_trial_sequences(trials, epoch_table, feature_cols):
    """
    make trial-level sequences per subject (300 main trials each: 30 epochs × 10 trials).
    x: (n_subjects, num_trials, n_features) (a subject's full trial sequence)
    y: (n_subjects, num_trials) (NaN everywhere except the last trial of each epoch,
       where y = rest_length for that epoch)
    lengths: (n_subjects,) — all 300 (uniform, kept for compatibility with pack_padded_sequence).
    """
    rest_dict = epoch_table.set_index(["subject_id", "epoch_num"])["rest_length"].to_dict()
    subject_ids = trials["subject_id"].unique()
    x_array, y_array = [], []

    for id in subject_ids:
        subject = trials[trials["subject_id"] == id].sort_values("trial_index").reset_index(drop=True)
        
        # get the feature values for all trials of the subj, change to numpy array
        x = subject[feature_cols].to_numpy(dtype=np.float32)

        # y with NaN everywhere
        y = np.full(len(subject), np.nan, dtype=np.float32)

        for epoch_num in subject["epoch_num"].unique():
            # last trial of the epoch
            last_idx = subject[subject["epoch_num"] == epoch_num].index[-1]
            rest_val = rest_dict.get((id, epoch_num), np.nan)
            if not np.isnan(rest_val):
                y[last_idx] = rest_val
        x_array.append(x)
        y_array.append(y)

    x = np.stack(x_array)
    y = np.stack(y_array)
    lengths = np.full(len(x), x.shape[1])
    return x, y, lengths