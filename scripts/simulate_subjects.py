"""
Simulate entire subjects (blockwise + trial-level)

Generates blockwise rows first (30 epochs per subject), then expands to 300 trials
per subject so that epoch-level stats match. 
Uses real training data to fit simple regression models and variability distributions.

gets called in run_all_models.py

Outputs:
  {out_dir}/simulated_blockwise.csv   - blockwise
  {out_dir}/simulated_main_trials.csv - trial-level

All models and stats are fit on training data only.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import (
    get_data,
    get_epoch_table,
    get_feature_columns,
    add_baseline_features,
    add_history_baseline_features,
)
from src.split import load_split_ids, train_test_split_pooled


################################################################################
# MODEL FITTING for later sims (all on training data only):
def fit_linear_and_sigmas(X, y, subject_ids, min_resid=0.01, min_subj=0.01):
    """Fit LinearRegression 
    return (model, resid_std, sigma_subject)."""
    model = LinearRegression().fit(X, y)
    resid = y - model.predict(X)
    subj = pd.Series(resid, index=subject_ids).groupby(level=0).mean()
    return model, max(float(np.std(resid)), min_resid), max(float(np.std(subj)), min_subj)


def fit_rest_model(df, feature_cols):
    """
    Ridge regression for rest_length. 
    Returns (model, scaler, resid_std, sigma_subject)
    Later: change this to be a glmer? to maintain within subject consistency more
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["rest_length"].to_numpy()

    # scale
    scaler = StandardScaler()
    model = Ridge(alpha=10.0).fit(scaler.fit_transform(X), y)
    residuals = y - model.predict(scaler.transform(X))

    # assign means and keep their residuals
    subject_means = df.assign(_r=residuals).groupby("subject_id")["_r"].mean()
    return model, scaler, max(float(np.std(residuals)), 0.5), max(float(np.std(subject_means)), 0.5)


def fit_performance_models(epoch_table):
    """
    Simple linear regressions for accuracy and RT. 
    accuracy: epoch, block, previous accuracy (get rid of block later?), game type
    rt: epoch, block,  game type
    Returns dict of models and noise parameters.
    """
    df = epoch_table.copy()
    is_digit_span = (df["game_type"].str.strip().str.lower() == "digit_span").astype(int).to_numpy()

    # fit acc model
    X_acc = np.column_stack([df["epoch_num"], df["block_num"], is_digit_span, df["accuracy_prev"].fillna(0.5)])
    acc_model, acc_resid, acc_sigma = fit_linear_and_sigmas(X_acc, df["avg_epoch_accuracy"].to_numpy(), df["subject_id"], 0.01, 0.01)

    # fit rt model
    valid = df["avg_rt"].notna()
    X_rt = np.column_stack([df.loc[valid, "epoch_num"], df.loc[valid, "block_num"], is_digit_span[valid]])
    rt_model, rt_resid, rt_sigma = fit_linear_and_sigmas(X_rt, df.loc[valid, "avg_rt"].to_numpy(), df.loc[valid, "subject_id"], 1.0, 1.0)

    return {
        "acc_model": acc_model, "acc_resid_std": acc_resid, "acc_sigma_subject": acc_sigma,
        "rt_model": rt_model, "rt_resid_std": rt_resid, "rt_sigma_subject": rt_sigma,
    }


def game_type_sequences(epoch_table):
    """
    Gets the two default orderings of game type (these are pre-set from the exp data)
    """
    def seq(group):
        return tuple(group.sort_values("epoch_num")["game_type"].str.strip().str.lower())
    seqs = epoch_table.groupby("subject_id").apply(seq, include_groups=False)
    seq_a = np.array(seqs.mode().iloc[0])
    seq_b = np.where(seq_a == "digit_span", "spatial_recall", "digit_span")
    return seq_a, seq_b


def real_data_dist_stats(epoch_table):
    """Get within-epoch variability and timeout distribution for each game type."""
    gt = epoch_table["game_type"].str.strip().str.lower()
    stats = {
        **{f"acc_sd_mean_{g}": epoch_table.loc[gt == g, "accuracy_sd"].mean() for g in ("digit_span", "spatial_recall")},
        **{f"rt_sd_mean_{g}": epoch_table.loc[gt == g, "rt_sd"].mean() for g in ("digit_span", "spatial_recall")},
        "timeout_values": epoch_table["num_timeouts"].dropna().astype(int).to_numpy(),
    }
    return stats


# do the fitting to get params we will need for future blockwise simulation
def fit_simulator(df_ext, epoch_table, feature_cols, epoch_table_for_stats=None):
    """Fit all simulation models on training data. Returns a config dict for generate_blockwise.
    """
    rest_model, rest_scaler, rest_resid_std, rest_sigma = fit_rest_model(df_ext, feature_cols)
    performance = fit_performance_models(epoch_table)

    stats_table = epoch_table_for_stats if epoch_table_for_stats is not None else epoch_table
    seq_a, seq_b = game_type_sequences(stats_table)
    stats = real_data_dist_stats(stats_table)

    return {
        "feature_cols": feature_cols,
        "rest_model": rest_model,
        "rest_scaler": rest_scaler,
        "rest_resid_std": rest_resid_std,
        "rest_sigma_subject": rest_sigma,
        **performance,
        **stats,
        "seq_a": seq_a,
        "seq_b": seq_b,
    }

def cue_for_epoch(epoch_num, game_type, prev_game_type):
    """Cue transition type from epoch position within the block structure."""
    if epoch_num == 1:
        return "stay_within_block"
    if (epoch_num - 1) % 3 == 0:
        return "switch_between_block" if game_type != prev_game_type else "stay_between_block"
    return "stay_within_block"

################################################################################
# Actually generate the blockwise data:

def generate_blockwise(n_subjects, sim, rng):
    """Generate n_subjects simulated subjects with 30 epochs each.

    Each subject gets a random intercept for rest, accuracy, and RT.
    Features are built sequentially so each epoch only uses information
    from previous epochs
    """
    feature_cols = sim["feature_cols"]
    rows = []

    for sid in range(1, n_subjects + 1):
        game_type_sequence = sim["seq_b"] if rng.random() < 0.5 else sim["seq_a"]
        intercept_rest = rng.normal(0, sim["rest_sigma_subject"])
        intercept_accuracy = rng.normal(0, sim["acc_sigma_subject"])
        intercept_rt = rng.normal(0, sim["rt_sigma_subject"])

        rests_so_far = 0
        rest_prev = np.nan
        accuracy_prev = np.nan
        rt_prev = np.nan
        game_type_prev = None
        cumulative_accuracy = {"digit_span": [], "spatial_recall": []}

        # go through every epoch:
        for epoch_idx in range(30):
            epoch_num = epoch_idx + 1
            block_num = (epoch_num - 1) // 3 + 1
            game_type = game_type_sequence[epoch_idx]
            is_digitspan = game_type == "digit_span"
            cue = cue_for_epoch(epoch_num, game_type, game_type_prev)

            # Accuracy
            accuracy_prev_fill = 0.5 if np.isnan(accuracy_prev) else accuracy_prev
            accuracy = float(
                sim["acc_model"].predict([[epoch_num, block_num, int(is_digitspan), accuracy_prev_fill]])[0]
                + intercept_accuracy + rng.normal(0, sim["acc_resid_std"])
            )
            accuracy = float(np.clip(accuracy, 0, 1))

            # Reaction time
            rt = float(
                sim["rt_model"].predict([[epoch_num, block_num, int(is_digitspan)]])[0]
                + intercept_rt + rng.normal(0, sim["rt_resid_std"])
            )
            rt = max(rt, 100.0)
            rt_prev_fill = 2000.0 if np.isnan(rt_prev) else rt_prev


            # Within-epoch variability
            accuracy_sd = float(rng.uniform(0.05, max(sim[f"acc_sd_mean_{game_type}"] * 1.5, 0.1)))
            rt_sd = float(rng.uniform(50, max(sim[f"rt_sd_mean_{game_type}"] * 1.5, 100)))
            num_timeouts = int(rng.choice(sim["timeout_values"]))

            # Rest-length prediction
            rest_prev_fill = 10.0 if np.isnan(rest_prev) else rest_prev

            if epoch_num == 1:
                prev_cue_stay = prev_cue_switch = 0
            else:
                prev_cue = cue_for_epoch(epoch_num - 1, game_type_sequence[epoch_idx - 1], game_type_sequence[epoch_idx - 2] if epoch_idx >= 2 else None)
                prev_cue_stay = int(prev_cue == "stay_between_block")
                prev_cue_switch = int(prev_cue == "switch_between_block")

            avg_ds = float(np.mean(cumulative_accuracy["digit_span"])) if cumulative_accuracy["digit_span"] else 0.5
            avg_sr = float(np.mean(cumulative_accuracy["spatial_recall"])) if cumulative_accuracy["spatial_recall"] else 0.5

            feat = {
                "epoch_num": epoch_num,
                "avg_epoch_accuracy": accuracy,
                "accuracy_sd": accuracy_sd,
                "avg_rt": rt,
                "rt_sd": rt_sd,
                "num_timeouts": num_timeouts,
                "game_type_digit_span": int(is_digitspan),
                "cue_stay_between_block": int(cue == "stay_between_block"),
                "cue_switch_between_block": int(cue == "switch_between_block"),
                "rests_taken_so_far": rests_so_far,
                "avg_accuracy_until_now_digit_span": avg_ds,
                "avg_accuracy_until_now_spatial_recall": avg_sr,
                "rest_length_prev": rest_prev_fill,
                "accuracy_prev": accuracy_prev_fill,
                "rt_prev": rt_prev_fill,
                "game_type_digit_span_prev": int(game_type_prev == "digit_span"),
                "previous_cue_stay_between_block": prev_cue_stay,
                "previous_cue_switch_between_block": prev_cue_switch,
            }
            for col in feature_cols:
                if col not in feat:
                    feat[col] = 0

            x = np.array([[feat[col] for col in feature_cols]], dtype=float)
            rest_pred = sim["rest_model"].predict(sim["rest_scaler"].transform(x))[0] + intercept_rest
            rest_length = int(np.clip(np.round(rest_pred + rng.normal(0, sim["rest_resid_std"])), 1, 20))

            cumulative_accuracy[game_type].append(accuracy)
            rows.append({
                "subject_id": f"simulated_{sid}",
                "subj_id": sid,
                "epoch_num": epoch_num,
                "block_num": block_num,
                "avg_epoch_accuracy": accuracy,
                "accuracy_sd": accuracy_sd,
                "avg_rt": rt,
                "rt_sd": rt_sd,
                "num_timeouts": num_timeouts,
                "game_type": game_type,
                "cue_transition_type": cue,
                "num_rest_in_chunk": rest_length,
                "rest_length": rest_length,
                "dataset": "simulated",
            })
            rests_so_far += rest_length
            rest_prev = rest_length
            accuracy_prev = accuracy
            rt_prev = rt
            game_type_prev = game_type

    return pd.DataFrame(rows)


def fit_simulator_and_generate_extended_and_blockwise(
    df_ext, epoch_table, ext_cols, n_subjects, rng, epoch_table_30=None,
):
    """
    does same generation but also for extended
    """
    if n_subjects <= 0:
        return pd.DataFrame(columns=df_ext.columns), pd.DataFrame()

    sim = fit_simulator(df_ext, epoch_table, ext_cols, epoch_table_for_stats=epoch_table_30)
    blockwise = generate_blockwise(n_subjects, sim, rng)
    blockwise_30 = blockwise.copy()
    blockwise = add_baseline_features(blockwise)
    blockwise = add_history_baseline_features(blockwise)
    blockwise = blockwise.dropna(subset=ext_cols)
    return blockwise, blockwise_30

################################################################################
# Expand blocks into trials for LSTM:

def blockwise_to_trials(blockwise, rng):
    """Expand blockwise to trial-level
    10 trials per epoch
    """
    rows = []
    for _, group in blockwise.groupby("subject_id"):
        group = group.sort_values("epoch_num")
        trial_index = 0
        rest_used_so_far = 0

        # go through each epoch
        for _, row in group.iterrows():
            accuracy = row["avg_epoch_accuracy"]
            rt_mean = row["avg_rt"]
            rt_sd = max(row["rt_sd"], 1)
            num_timeouts = int(row["num_timeouts"])
            rest_length = int(row["rest_length"])
            trial_type = "ds_main_response" if row["game_type"] == "digit_span" else "sr_main_response"

            corrects = rng.binomial(1, accuracy, size=10)
            rts = np.maximum(rng.normal(rt_mean, rt_sd, size=10), 100)
            timeout_idx = np.array([], dtype=np.intp)
            if num_timeouts > 0:
                timeout_idx = rng.choice(10, size=min(num_timeouts, 10), replace=False)
                corrects[timeout_idx] = 0
                rts[timeout_idx] = 5000

            for k in range(10):
                trial_index += 1
                rows.append({
                    "subject_id": row["subject_id"],
                    "subj_id": row["subj_id"],
                    "dataset": "simulated",
                    "trial_type": trial_type,
                    "trial_index": trial_index,
                    "epoch_num": int(row["epoch_num"]),
                    "block_num": int(row["block_num"]),
                    "game_type": row["game_type"],
                    "cue_transition_type": row["cue_transition_type"],
                    "is_correct_numeric": float(corrects[k]),
                    "rt": float(rts[k]),
                    "timed_out": bool(k in timeout_idx),
                    "num_rest_in_chunk": rest_length,
                    "rest_used_so_far": rest_used_so_far,
                })
            rest_used_so_far += rest_length

    return pd.DataFrame(rows)


##################################################
def main():
    parser = argparse.ArgumentParser(description="Simulate subjects (blockwise + trial-level) only from train set.")
    parser.add_argument("--n-subjects", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="cleaned_exp_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-split", type=str, default="splits/pooled_80_20")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading existing data...")
    df = get_data()
    ext_cols = get_feature_columns(baseline_only=False, include_dataset=False)
    df_ext = df.dropna(subset=ext_cols)

    epoch_table = get_epoch_table()
    epoch_table = add_baseline_features(epoch_table)
    epoch_table = add_history_baseline_features(epoch_table)

    if args.load_split:
        train_ids, test_ids = load_split_ids(args.load_split)
        print(f"Using split from {args.load_split}: {len(train_ids)} train, {len(test_ids)} test subjects.")
    else:
        subj_df = epoch_table.drop_duplicates("subject_id")[["subject_id", "dataset"]]
        train_subj, test_subj = train_test_split_pooled(subj_df, test_frac=args.test_frac, random_state=args.seed)
        train_ids = set(train_subj["subject_id"])
        test_ids = set(test_subj["subject_id"])
        print(f"No split file provided. Split by subject (test_frac={args.test_frac}): {len(train_ids)} train, {len(test_ids)} test subjects.")

    df_ext = df_ext[df_ext["subject_id"].isin(train_ids)].copy()
    epoch_table = epoch_table[epoch_table["subject_id"].isin(train_ids)].copy()
    print(f"Fitting on TRAIN only ({len(train_ids)} subjects, {len(epoch_table)} epochs)...")

    sim = fit_simulator(df_ext, epoch_table, ext_cols)
    print(f"Counterbalancing: epoch 1 = {sim['seq_a'][0]} or {sim['seq_b'][0]}")

    print(f"Generating {args.n_subjects} subjects (blockwise)...")
    blockwise = generate_blockwise(args.n_subjects, sim, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blockwise_path = out_dir / "simulated_blockwise.csv"
    blockwise.to_csv(blockwise_path, index=False)
    print(f"Saved {blockwise_path} ({len(blockwise)} rows)")

    print("Expanding to trial-level...")
    trials = blockwise_to_trials(blockwise, rng)
    trials_path = out_dir / "simulated_main_trials.csv"
    trials.to_csv(trials_path, index=False)
    print(f"Saved {trials_path} ({len(trials)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()