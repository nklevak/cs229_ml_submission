"""
Run baselines (Ridge + GBM with tuning) with + without history
 and LSTM on dataset and/or pooled splits.
Can also add simulated subjects to the pooled train set by editing --sim-multiplier > 0

multiplier = how many more simulated subjects to add relative to initial dataset
so 1x sim = sim same number as in the original training set

copy paste these:
uv run python scripts/run_all_models.py --models-dir models/mar13
uv run python scripts/run_all_models.py --sim-multiplier 1 --models-dir models/mar13_1xsimulated
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scripts.simulate_subjects import fit_simulator_and_generate_extended_and_blockwise, blockwise_to_trials
from src.preprocess import (
    MAIN_RESPONSE_TRIAL_TYPES,
    get_data,
    get_epoch_table,
    get_feature_columns,
    get_trial_feature_columns,
    get_trials,
    add_trial_features,
)
from src.split import (
    kfold_by_subject,
    load_split_ids,
    save_split_ids,
    split_by_dataset,
    split_df_by_subject_ids,
    train_test_split_pooled,
)
from scripts.train_lstm import run_lstm


################################################################################
# TUNING:

N_FOLDS = 15
ALPHAS_STR = "0.01,0.05,0.1,0.5,1,10,20,100"
GBM_N_ESTIMATORS_STR = "100,200,300,400,500"
GBM_MAX_DEPTH_STR = "1,2,3,4"
GBM_LEARNING_RATE = 0.05 # i tried different onces but this was best (compared to 0.1)


################################################################################
# Helper functions for baselines:

def folds_to_arrays(train_dfs, val_dfs, cols):
    """Convert fold DataFrames to (X_train, y_train, X_val, y_val) numpy tuples."""
    out = []
    for train_fold, val_fold in zip(train_dfs, val_dfs):
        out.append((
            train_fold[cols].to_numpy().astype(float), train_fold["rest_length"].to_numpy(),
            val_fold[cols].to_numpy().astype(float), val_fold["rest_length"].to_numpy(),
        ))
    return out


def sweep_ridge_alpha(folds, alphas, seed=42):
    """For each given fold fits Ridge on training set and tests on val set
    calculates the MAE and r squared per alpha
    returns results list of alpha, mae mean, r2 mean
    """
    results = []
    for alpha in alphas:
        mae_list, r2_list = [], []
        for X_tr, y_tr, X_val, y_val in folds:

            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)

            # fit model on the train set:
            model = Ridge(alpha=alpha, random_state=seed).fit(X_tr, y_tr)
            pred = model.predict(X_val) # predict on the val set (for each fold)
            mae_list.append(mean_absolute_error(y_val, pred))
            r2_list.append(r2_score(y_val, pred))

        results.append({"alpha": alpha, "mae_mean": np.mean(mae_list), "r2_mean": np.mean(r2_list)})
    return results


def make_gbm_params(n_est, max_d, lr, seed):
    """returns a dict of params to feed into the GBM
    putting into a function to maintain consistency
    has early stopping and uses 10% of val to decide when to stop
    """
    return dict(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=lr,
        random_state=seed,
        validation_fraction=0.1,
        n_iter_no_change=10,
        # tol=1e-4, # default in sklearn ?
    )


def tune_gbm(folds, n_est_list, max_d_list, seed=42, lr=GBM_LEARNING_RATE):
    """
    For each depth and n_estimator combo,
    and each given fold, fits gbm on training set and tests on val set
    calculates the r squared for that combo
    returns best n_estimator and depth by mean r squared
    default lr of 0.05 (GBM_LEARNING_RATE)
    """
    best_r2, best_n, best_d = -np.inf, n_est_list[0], max_d_list[0]
    for n_est in n_est_list:
        for d in max_d_list:
            r2_list = []
            for X_tr, y_tr, X_val, y_val in folds:
                model = GradientBoostingRegressor(**make_gbm_params(n_est, d, lr, seed)).fit(X_tr, y_tr)
                r2_list.append(r2_score(y_val, model.predict(X_val)))
            mean_r2 = float(np.mean(r2_list))
            if mean_r2 > best_r2:
                best_r2, best_n, best_d = mean_r2, n_est, d
    return best_n, best_d


################################################################################
# RUN THE BASELINES (RIDGE + GBM) with and without history:
# For a given train/test split: 
# tune Ridge + GBM on train folds
# fit on full train
# evaluate on test
# save models
# extended = with history (the same subjects as baseline but with the added features)

def run_baselines_for_split(
    split_name,
    train_baseline,
    test_baseline,
    train_extended,
    test_extended,
    baseline_cols,
    ext_cols,
    save_path,
    seed=42,
):
    """
    Tune Ridge/GBM on train folds, then fit on full train, then evaluate on test and save results + models
    Returns dict with data split info, mae, r2, and the tuned params.
    both history and baseline
    """
    # make the folds for baseline + extended 
    folds_baseline = list(kfold_by_subject(train_baseline, n_splits=N_FOLDS, random_state=seed))
    folds_ext = list(kfold_by_subject(train_extended, n_splits=N_FOLDS, random_state=seed))
    folds_baseline_arr = folds_to_arrays([f[0] for f in folds_baseline], [f[1] for f in folds_baseline], baseline_cols)
    folds_ext_arr = folds_to_arrays([f[0] for f in folds_ext], [f[1] for f in folds_ext], ext_cols)

    # tune ridge and print
    alphas = [float(x) for x in ALPHAS_STR.split(",")]
    ridge_alpha_baseline = max(sweep_ridge_alpha(folds_baseline_arr, alphas, seed=seed), key=lambda r: r["r2_mean"])["alpha"]
    ridge_alpha_ext = max(sweep_ridge_alpha(folds_ext_arr, alphas, seed=seed), key=lambda r: r["r2_mean"])["alpha"]
    print(f"Tuned Ridge alphas: baseline={ridge_alpha_baseline}, extended with hist={ridge_alpha_ext}")

    # tune gbm and print tuned vals
    n_list = [int(x) for x in GBM_N_ESTIMATORS_STR.split(",")]
    d_list = [int(x) for x in GBM_MAX_DEPTH_STR.split(",")]
    gbm_n_baseline, gbm_d_baseline = tune_gbm(folds_baseline_arr, n_list, d_list, seed=seed, lr=GBM_LEARNING_RATE)
    gbm_n_ext, gbm_d_ext = tune_gbm(folds_ext_arr, n_list, d_list, seed=seed, lr=GBM_LEARNING_RATE)
    print(f"Tuned GBM: baseline n={gbm_n_baseline} d={gbm_d_baseline}, extended with hist n={gbm_n_ext} d={gbm_d_ext}")

    # make x and y for the train and test sets
    X_train_baseline = train_baseline[baseline_cols].to_numpy().astype(float)
    y_train_baseline = train_baseline["rest_length"].to_numpy()
    X_test_baseline = test_baseline[baseline_cols].to_numpy().astype(float)
    y_test_baseline = test_baseline["rest_length"].to_numpy()

    X_train_ext = train_extended[ext_cols].to_numpy().astype(float)
    y_train_ext = train_extended["rest_length"].to_numpy()
    X_test_ext = test_extended[ext_cols].to_numpy().astype(float)
    y_test_ext = test_extended["rest_length"].to_numpy()

    X_test_baseline_unscaled = X_test_baseline.copy()
    X_test_ext_unscaled = X_test_ext.copy()

    # Scale for Ridge
    scaler_baseline = StandardScaler().fit(X_train_baseline)
    X_train_baseline = scaler_baseline.transform(X_train_baseline)
    X_test_baseline = scaler_baseline.transform(X_test_baseline)
    scaler_ext = StandardScaler().fit(X_train_ext)
    X_train_ext = scaler_ext.transform(X_train_ext)
    X_test_ext = scaler_ext.transform(X_test_ext)

    # Fit ridge
    ridge_baseline = Ridge(alpha=ridge_alpha_baseline, random_state=seed).fit(X_train_baseline, y_train_baseline)
    ridge_ext = Ridge(alpha=ridge_alpha_ext, random_state=seed).fit(X_train_ext, y_train_ext)
    
    # fit gbm
    gbm_baseline = GradientBoostingRegressor(**make_gbm_params(gbm_n_baseline, gbm_d_baseline, GBM_LEARNING_RATE, seed)).fit(
        train_baseline[baseline_cols].to_numpy().astype(float), y_train_baseline
    )
    gbm_ext = GradientBoostingRegressor(**make_gbm_params(gbm_n_ext, gbm_d_ext, GBM_LEARNING_RATE, seed)).fit(
        train_extended[ext_cols].to_numpy().astype(float), y_train_ext
    )

    # save the models
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = str(path)
    joblib.dump((ridge_baseline, scaler_baseline), f"{base}_{split_name}_ridge_baseline.pkl")
    joblib.dump((ridge_ext, scaler_ext), f"{base}_{split_name}_ridge_extended.pkl")
    joblib.dump(gbm_baseline, f"{base}_{split_name}_gbm_baseline.pkl")
    joblib.dump(gbm_ext, f"{base}_{split_name}_gbm_extended.pkl")
    print(f"Models saved in: {base}_{split_name}_*.pkl")

    return {
        "split": split_name,
        "ridge_baseline_mae": mean_absolute_error(y_test_baseline, ridge_baseline.predict(X_test_baseline)),
        "ridge_baseline_r2": r2_score(y_test_baseline, ridge_baseline.predict(X_test_baseline)),
        "gbm_baseline_mae": mean_absolute_error(y_test_baseline, gbm_baseline.predict(X_test_baseline_unscaled)),
        "gbm_baseline_r2": r2_score(y_test_baseline, gbm_baseline.predict(X_test_baseline_unscaled)),
        "ridge_extended_mae": mean_absolute_error(y_test_ext, ridge_ext.predict(X_test_ext)),
        "ridge_extended_r2": r2_score(y_test_ext, ridge_ext.predict(X_test_ext)),
        "gbm_extended_mae": mean_absolute_error(y_test_ext, gbm_ext.predict(X_test_ext_unscaled)),
        "gbm_extended_r2": r2_score(y_test_ext, gbm_ext.predict(X_test_ext_unscaled)),
        "ridge_alpha_baseline": ridge_alpha_baseline,
        "ridge_alpha_extended": ridge_alpha_ext,
        "gbm_n_baseline": gbm_n_baseline,
        "gbm_d_baseline": gbm_d_baseline,
        "gbm_n_extended": gbm_n_ext,
        "gbm_d_extended": gbm_d_ext,
        "baseline_cols": list(baseline_cols),
        "ext_cols": list(ext_cols),
    }

##############################################################################################
# data prep before calling function above: (this data will get passed in to the training loop later)

# no sim: two splits (dataset = train original / test replication; pooled = 80/20 by subject).
def get_splits_standard(args, df_baseline, df_extended):
    """
    Takes in command line args, and the entire baseline and extended dfs
    Returns list of split_name (pooled/dataset), train_baseline data, test_baseline data, 
    train_extended data, test_extended data, baseline_cols features, ext_cols features
    """
    load_premade_pooled_split = args.load_split # filename leading to the desired split json
    seed = args.seed

    splits_out = []
    for split_name in ["dataset", "pooled"]:

        baseline_cols = get_feature_columns(baseline_only=True, include_dataset=(split_name == "pooled"))
        ext_cols = get_feature_columns(baseline_only=False, include_dataset=(split_name == "pooled"))
        
        if split_name == "dataset":#for dataset just make original = train and rep = test
            tr_baseline, te_baseline = split_by_dataset(df_baseline, "original", "replication")
            tr_ext, te_ext = split_by_dataset(df_extended, "original", "replication")
        else:
            if (load_premade_pooled_split != ""):
                train_ids, test_ids = load_split_ids(load_premade_pooled_split)
                tr_baseline, te_baseline = split_df_by_subject_ids(df_baseline, train_ids, test_ids)
                tr_ext, te_ext = split_df_by_subject_ids(df_extended, train_ids, test_ids)
            else:
                tr_baseline, te_baseline = train_test_split_pooled(df_baseline, test_frac=args.test_frac, random_state=seed)
                tr_ext, te_ext = train_test_split_pooled(df_extended, test_frac=args.test_frac, random_state=seed)
        
        splits_out.append((split_name, tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols))
    
    return splits_out

# simulated subjects added to train; only does pooled split
def get_splits_with_simulation(args, models_dir, df_baseline, df_extended):
    """
    Add simulated subjects to training data and split (as above) but only for pooled
    """
    rng = np.random.default_rng(args.seed)
    ext_cols = get_feature_columns(baseline_only=False, include_dataset=True)
    baseline_cols = get_feature_columns(baseline_only=True, include_dataset=True)
    train_ids, test_ids = load_split_ids(args.load_split)
    n_train = len(train_ids)
    n_sim = max(0, int(round(args.sim_multiplier * n_train)))
    print(f"Number of training subjects: {n_train}: adding {n_sim} simulated subjects ({args.sim_multiplier}x)")

    if n_sim == 0:
        tr_baseline, te_baseline = split_df_by_subject_ids(df_baseline, train_ids, test_ids)
        tr_ext, te_ext = split_df_by_subject_ids(df_extended, train_ids, test_ids)
        split_path = args.load_split
        epoch_full = get_epoch_table()
        real_trials = get_trials()
        real_trials = real_trials[real_trials["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)].copy()
        real_trials = add_trial_features(real_trials, epoch_full)
        lstm_extra = {
            "trials_df": real_trials.dropna(subset=get_trial_feature_columns(include_dataset=True)),
            "epoch_table_df": epoch_full,
        }
    else:
        train_extended = split_df_by_subject_ids(df_extended, train_ids, test_ids)[0]
        epoch_full = get_epoch_table()
        epoch_train = epoch_full[epoch_full["subject_id"].isin(train_ids)].copy()
        sim_extended, sim_blockwise = fit_simulator_and_generate_extended_and_blockwise(
            train_extended, train_extended, ext_cols, n_sim, rng, epoch_table_30=epoch_train
        )
        df_baseline_aug = pd.concat([df_baseline, sim_extended], ignore_index=True)
        df_ext_aug = pd.concat([df_extended, sim_extended], ignore_index=True)
        sim_ids = [f"simulated_{i}" for i in range(1, n_sim + 1)]
        train_ids_aug = sorted(set(train_ids) | set(sim_ids))
        save_split_ids(train_ids_aug, list(test_ids), models_dir / "split_augmented.json")
        split_path = str(models_dir / "split_augmented.json")
        tr_baseline, te_baseline = split_df_by_subject_ids(df_baseline_aug, train_ids_aug, test_ids)
        tr_ext, te_ext = split_df_by_subject_ids(df_ext_aug, train_ids_aug, test_ids)

        # LSTM inputs with simulated data:
        real_trials = get_trials()
        real_trials = real_trials[real_trials["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)].copy()
        real_trials = add_trial_features(real_trials, epoch_full)
        sim_trials = blockwise_to_trials(sim_blockwise, rng)
        sim_trials = add_trial_features(sim_trials, epoch_table=sim_blockwise)
        trials_aug = pd.concat([real_trials, sim_trials], ignore_index=True)
        epoch_aug = pd.concat([
            epoch_full[["subject_id", "epoch_num", "rest_length"]],
            sim_blockwise[["subject_id", "epoch_num", "rest_length"]],
        ], ignore_index=True)
        lstm_extra = {"trials_df": trials_aug.dropna(subset=get_trial_feature_columns(include_dataset=True)), "epoch_table_df": epoch_aug}

    return [("pooled", tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols)], split_path, lstm_extra


################################################################################
# Plots for interpretation:

def plot_mae_r2(baseline_results, lstm_results, out_path):
    """Make basic plots for comparison:
    comparison_mae.png and comparison_r2.png
    for baselines, extended baselines, and LSTM"""
    labels = ["Ridge\n(baseline)", "GBM\n(baseline)", "Ridge\n(extended)", "GBM\n(extended)", "LSTM"]
    keys_mae = ["ridge_baseline_mae", "gbm_baseline_mae", "ridge_extended_mae", "gbm_extended_mae"]
    keys_r2 = ["ridge_baseline_r2", "gbm_baseline_r2", "ridge_extended_r2", "gbm_extended_r2"]
    x = np.arange(len(labels))
    width = 0.35

    for metric, keys, ylabel, fname in [
        ("mae", keys_mae, "Test MAE", "comparison_mae.png"),
        ("r2", keys_r2, "Test R²", "comparison_r2.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, res in enumerate(baseline_results):
            vals = [res[k] for k in keys]
            lstm_r = next((r for r in lstm_results if r["split"] == res["split"]), None)
            vals.append(lstm_r[ f"lstm_{metric}"] if lstm_r else np.nan)
            leg = "train=original, test=replication" if res["split"] == "dataset" else "pooled (80% train, 20% test)"
            ax.bar(x + (i - 0.5) * width, vals, width, label=leg)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / fname, dpi=150)
        plt.close(fig)

# document the params we ended up using + num simulated etc
def print_params(models_dir, baseline_results, lstm_results, extra_args=None):
    """Save tuned values and params in tuned_params.md"""
    extra_args = extra_args or {}
    lines = [
        "# Tuned values and params",
        "",
        "## Tuning search",
        f"- Ridge alphas: {extra_args.get('alphas', ALPHAS_STR)}",
        f"- GBM n_estimators: {extra_args.get('gbm_n', GBM_N_ESTIMATORS_STR)}",
        f"- GBM max_depth: {extra_args.get('gbm_d', GBM_MAX_DEPTH_STR)}",
        f"- n_folds: {N_FOLDS}",
        "",
        "## Run",
        f"- load_split: {extra_args.get('load_split', '—')}",
        f"- simulated: {extra_args.get('simulated', '—')}",
        f"- seed: {extra_args.get('seed', '—')}",
        "",
    ]
    for res in baseline_results:
        lines.append(f"## {res['split']} split (chosen values)")
        lines.append(f"- Ridge alpha baseline: {res.get('ridge_alpha_baseline')}, extended: {res.get('ridge_alpha_extended')}")
        lines.append(f"- GBM n/d baseline: {res.get('gbm_n_baseline')}/{res.get('gbm_d_baseline')}, extended: {res.get('gbm_n_extended')}/{res.get('gbm_d_extended')}")
        lines.append("")
    lines.append("## LSTM")
    if lstm_results:
        for k, v in lstm_results[0].get("params", {}).items():
            lines.append(f"- {k}: {v}")
    (models_dir / "tuned_params.md").write_text("\n".join(lines))


################################################################################
# MAIN PIPELINE:

def run(args, models_dir):
    """Run Ridge + GBM + LSTM baselines + extended
    If sim_multiplier > 0, use only pooled split with the added simulated train data
    """
    df = get_data()
    with_simulation = args.sim_multiplier > 0

    if with_simulation:
        ext_cols = get_feature_columns(baseline_only=False, include_dataset=True)
        baseline_cols = get_feature_columns(baseline_only=True, include_dataset=True)
        df_baseline = df.dropna(subset=baseline_cols)
        df_extended = df.dropna(subset=ext_cols)
        splits_list, split_path, lstm_extra = get_splits_with_simulation(args, models_dir, df_baseline, df_extended)
        (split_name, tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols) = splits_list[0]

        print("\nRUNNING BASELINES (Ridge + GBM)\n")
        res = run_baselines_for_split(
            split_name, tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols,
            save_path=str(models_dir / "baselines"), seed=args.seed,
        )
        print(f"Test MAE Ridge baseline: {res['ridge_baseline_mae']:.4f}  GBM baseline: {res['gbm_baseline_mae']:.4f}  Ridge ext: {res['ridge_extended_mae']:.4f}  GBM ext: {res['gbm_extended_mae']:.4f}")
        print(f"Test R² Ridge baseline: {res['ridge_baseline_r2']:.4f}  GBM baseline: {res['gbm_baseline_r2']:.4f}  Ridge ext: {res['ridge_extended_r2']:.4f}  GBM ext: {res['gbm_extended_r2']:.4f}")

        print("\nRUNNING LSTM\n")
        lstm_kw = dict(split="pooled", load_split=split_path, test_frac=args.test_frac, seed=args.seed, save=str(models_dir / "lstm_pooled.pt"))
        lstm_kw.update(lstm_extra)
        lstm_res = run_lstm(**lstm_kw)
        print(f"Test MAE: {lstm_res['lstm_mae']:.4f}, R²: {lstm_res['lstm_r2']:.4f}")

        n_train = len(load_split_ids(args.load_split)[0])
        n_sim = max(0, int(round(args.sim_multiplier * n_train)))
        baseline_results = [res]
        lstm_results = [lstm_res]
        results_json = [{
            "split": "pooled",
            "ridge_baseline_mae": res["ridge_baseline_mae"], "ridge_baseline_r2": res["ridge_baseline_r2"],
            "gbm_baseline_mae": res["gbm_baseline_mae"], "gbm_baseline_r2": res["gbm_baseline_r2"],
            "ridge_extended_mae": res["ridge_extended_mae"], "ridge_extended_r2": res["ridge_extended_r2"],
            "gbm_extended_mae": res["gbm_extended_mae"], "gbm_extended_r2": res["gbm_extended_r2"],
            "lstm_mae": lstm_res["lstm_mae"], "lstm_r2": lstm_res["lstm_r2"], "lstm_params": lstm_res["params"],
            "n_train_subjects": n_train, "n_simulated_subjects": n_sim, "sim_multiplier": args.sim_multiplier,
        }]
        extra_args = {"load_split": args.load_split, "simulated": f"{n_sim} simulated subjects", "seed": args.seed}
    else: # no additional simulated subjects
        df_baseline = df.dropna(subset=get_feature_columns(baseline_only=True))
        df_extended = df.dropna(subset=get_feature_columns(baseline_only=False))
        splits = get_splits_standard(args, df_baseline, df_extended)
        baseline_results = []
        for (split_name, tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols) in splits:
            print(f"--- Split: {split_name} ---")
            res = run_baselines_for_split(
                split_name, tr_baseline, te_baseline, tr_ext, te_ext, baseline_cols, ext_cols,
                save_path=str(models_dir / "baselines"), seed=args.seed,
            )
            baseline_results.append(res)
            print(f"Test MAE  Ridge baseline: {res['ridge_baseline_mae']:.4f}  GBM baseline: {res['gbm_baseline_mae']:.4f}  Ridge ext: {res['ridge_extended_mae']:.4f}  GBM ext: {res['gbm_extended_mae']:.4f}")
            print(f"Test R²   Ridge baseline: {res['ridge_baseline_r2']:.4f}  GBM baseline: {res['gbm_baseline_r2']:.4f}  Ridge ext: {res['ridge_extended_r2']:.4f}  GBM ext: {res['gbm_extended_r2']:.4f}")

        print("\nRUNNING LSTM\n")
        lstm_results = []
        for split_name in ["dataset", "pooled"]:
            print(f"\n--- LSTM: {split_name} ---")
            load_split_path = args.load_split if split_name == "pooled" else None
            lstm_res = run_lstm(split=split_name, load_split=load_split_path, test_frac=args.test_frac, seed=args.seed, save=str(models_dir / f"lstm_{split_name}.pt"))
            lstm_results.append(lstm_res)
            print(f"  Test MAE: {lstm_res['lstm_mae']:.4f}, R²: {lstm_res['lstm_r2']:.4f}")

        results_json = [
            {**{k: b[k] for k in ["split", "ridge_baseline_mae", "ridge_baseline_r2", "gbm_baseline_mae", "gbm_baseline_r2", "ridge_extended_mae", "ridge_extended_r2", "gbm_extended_mae", "gbm_extended_r2"]}, "lstm_mae": l["lstm_mae"], "lstm_r2": l["lstm_r2"], "lstm_params": l["params"]}
            for b, l in zip(baseline_results, lstm_results)
        ]
        extra_args = {"load_split": args.load_split, "seed": args.seed}

    (models_dir / "results.json").write_text(json.dumps(results_json, indent=2))
    plot_mae_r2(baseline_results, lstm_results, models_dir)
    print_params(models_dir, baseline_results, lstm_results, extra_args=extra_args)
    print(f"\nSaved results, plots, tuned_params.md in {models_dir}")


################################################################################
# Main() and command line args

def main():
    parser = argparse.ArgumentParser(description="Run Ridge + GBM + LSTM. Use --sim-multiplier > 0 to add simulated train data.")
    parser.add_argument("--models-dir", type=str, default="models", help="Output directory for models and results")
    parser.add_argument("--load-split", type=str, default="splits/pooled_80_20", help="Pooled split path")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-multiplier", type=float, default=0.0, help="Simulated subjects = this × train size (0 = no sim). Runs pooled only.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    run(args, models_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()