"""
Additional plots: 
- no simulation pooled results vs 1x simulation pooled results (MAE / r squared)
- predicted versus actual distributions for the best performing models

command:
uv run python scripts/make_additional_plots.py

saved the plots in models/figs/mar13_comparisons (but can change this later when I have to make more plots that compare more different things)
"""
import json
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import (
    get_data,
    get_epoch_table,
    get_trials,
    add_trial_features,
    get_feature_columns,
    get_trial_feature_columns,
    build_trial_sequences,
    MAIN_RESPONSE_TRIAL_TYPES,
)
from src.split import load_split_ids
from src.lstm_model import RestLSTM

COMPARISON_COLORS = ["#2ca25f", "#99d8c9"]  # no sim, 1x sim

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR = MODELS_DIR / "figs" / "mar13_comparisons"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SUBJECTS_TO_PLOT = 3
seed = 42

# (models_dir, split_path, out_fname, title_suffix)
REST_VS_EPOCH_CONFIGS = [
    (MODELS_DIR / "mar13_nosim", PROJECT_ROOT / "splits" / "pooled_80_20.json", "rest_vs_epoch_nosim.png", "no simulated subjects"),
    (MODELS_DIR / "mar13_1xsim", MODELS_DIR / "mar13_1xsim" / "split_augmented.json", "rest_vs_epoch_1xsim.png", "1× simulated subjects"),
]

# Rest-vs-epoch line plot: legend labels and styling (same for all panels)
REST_VS_EPOCH_LABELS = ["real dist", "Ridge (+ hist)", "GBM (+ hist)", "LSTM"]
REST_VS_EPOCH_STYLE = {
    "dashes": {"real dist": (None, None), "Ridge (+ hist)": (2, 1), "GBM (+ hist)": (2, 1), "LSTM": (2, 1)},
    "palette": {"real dist": "black", "Ridge (+ hist)": "C0", "GBM (+ hist)": "C1", "LSTM": "C2"},
}

def load_pooled_results(models_dir):
    """Load results.json from a run in a given model directory"""
    path = models_dir / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for entry in data:
        if entry.get("split") == "pooled":
            return entry
    return None

def plot_no_sim_vs_1x_sim():
    """Bar chart 1x sim r squared and MAE for each model (pooled split).
    This is the same as the plot generated per individual run comparing pooled vs dataset mae and r squared
    but this time it's comparing simulation vs no simulation"""
    no_sim = load_pooled_results(MODELS_DIR / "mar13_nosim")
    one_sim = load_pooled_results(MODELS_DIR / "mar13_1xsim")

    model_labels = ["Ridge\n(baseline)", "GBM\n(baseline)", "Ridge\n(+ history)", "GBM\n(+ history)", "LSTM"]
    keys_mae = ["ridge_baseline_mae", "gbm_baseline_mae", "ridge_extended_mae", "gbm_extended_mae", "lstm_mae"]
    keys_r2 = ["ridge_baseline_r2", "gbm_baseline_r2", "ridge_extended_r2", "gbm_extended_r2", "lstm_r2"]

    x = np.arange(len(model_labels))
    width = 0.35
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for keys, ylabel, fname in [
        (keys_mae, "Test MAE", "no_sim_vs_1xsim_mae.png"),
        (keys_r2, "Test R²", "no_sim_vs_1xsim_r2.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals_nosim = [no_sim[k] for k in keys]
        vals_1sim = [one_sim[k] for k in keys]
        ax.bar(x - width / 2, vals_nosim, width, label="No simulated subjects", color=COMPARISON_COLORS[0])
        ax.bar(x + width / 2, vals_1sim, width, label="Augmented with 1x simulated subjects", color=COMPARISON_COLORS[1])
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.legend()
        ax.set_title(ylabel)
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT_DIR / fname}")


def load_lstm_predictions(trials_test_subjects, epoch_table, models_dir):
    """Load LSTM
    run on trials_test_subjects
    returns the lstm, the index of a given subject, and the number of vals corresponding t that subject
    """
    feature_cols = (models_dir / "lstm_pooled_features.txt").read_text().strip().splitlines()
    X, _, len_test = build_trial_sequences(trials_test_subjects, epoch_table, feature_cols)
    ckpt = torch.load(models_dir / "lstm_pooled.pt", map_location="cpu", weights_only=False)
    model = RestLSTM(
        n_features=X.shape[2],
        hidden_size=ckpt["hidden"],
        num_layers=ckpt["layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    X_scaled = (X.reshape(-1, X.shape[2]) - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    X_scaled = X_scaled.reshape(X.shape).astype(np.float32)
    with torch.no_grad():
        lstm_out = model(
            torch.from_numpy(X_scaled).float(),
            lengths=torch.from_numpy(len_test).long(),
        ).squeeze(-1).numpy()
    subject_to_idx = {s: i for i, s in enumerate(trials_test_subjects["subject_id"].unique())}
    return lstm_out, subject_to_idx, len_test


def predictions_for_one_subject(subj_epoch, ridge, scaler, gbm, ext_cols, lstm_out, len_test, subject_to_idx, sid):
    """Returns the epochs, actual dist, and predicted dist of each model"""
    actual = subj_epoch["rest_length"].to_numpy()
    n = len(actual)
    X = subj_epoch[ext_cols].to_numpy().astype(float)
    ridge_pred = ridge.predict(scaler.transform(X))[:n]
    gbm_pred = gbm.predict(X)[:n]
    idx = subject_to_idx.get(sid)
    if idx is None:
        lstm_pred = np.full(n, np.nan)
    else:
        L = int(len_test[idx])
        lstm_pred = lstm_out[idx, 9:L:10][:30]  # end-of-epoch predictions
        lstm_pred = np.resize(lstm_pred, n) if len(lstm_pred) < n else lstm_pred[:n]
    return np.arange(1, n + 1), actual, ridge_pred, gbm_pred, lstm_pred


def rest_vs_epoch_one_run(models_dir, split_path, out_fname, title_suffix):
    """
    Line plot rest length vs epoch for a few random test subjects 
    (real + Ridge/GBM/LSTM)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, test_ids = load_split_ids(split_path)
    test_ids = list(test_ids)

    df = get_data()
    ext_cols = get_feature_columns(baseline_only=False, include_dataset=True)
    test_ext = df[df["subject_id"].isin(test_ids)].dropna(subset=ext_cols)
    if len(test_ext) == 0:
        print(f"No test data for {title_suffix}; skipping")
        return

    ridge_path = models_dir / "baselines_pooled_ridge_extended.pkl"
    gbm_path = models_dir / "baselines_pooled_gbm_extended.pkl"
    lstm_path = models_dir / "lstm_pooled.pt"
    if not ridge_path.exists() or not gbm_path.exists() or not lstm_path.exists():
        print(f"Missing model files in {models_dir}; skipping")
        return
    ridge, scaler = joblib.load(ridge_path)
    gbm = joblib.load(gbm_path)

    epoch_table = get_epoch_table()
    trials = get_trials()
    trials = trials[trials["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)]
    trials = add_trial_features(trials, epoch_table)
    feature_cols = (models_dir / "lstm_pooled_features.txt").read_text().strip().splitlines()
    trials_test_subjects = trials[trials["subject_id"].isin(test_ids)].dropna(subset=feature_cols)
    if len(trials_test_subjects) == 0:
        print("No test trials for LSTM; skipping")
        return
    lstm_out, subject_to_idx, len_test = load_lstm_predictions(trials_test_subjects, epoch_table, models_dir)

    rng = np.random.default_rng(seed)
    n_pick = min(N_SUBJECTS_TO_PLOT, test_ext["subject_id"].nunique())
    chosen = rng.choice(test_ext["subject_id"].unique(), size=n_pick, replace=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, n_pick, figsize=(5 * n_pick, 4))
    axes = np.atleast_1d(axes).flatten()
    for ax, sid in zip(axes, chosen):
        subj = test_ext[test_ext["subject_id"] == sid].sort_values("epoch_num")
        x, actual, rp, gp, lp = predictions_for_one_subject(
            subj, ridge, scaler, gbm, ext_cols, lstm_out, len_test, subject_to_idx, sid
        )
        plot_df = pd.DataFrame({
            "epoch": np.tile(x, 4),
            "rest length": np.concatenate([actual, rp, gp, lp]),
            "model": np.repeat(REST_VS_EPOCH_LABELS, len(x)),
        })
        sns.lineplot(data=plot_df, x="epoch", y="rest length", hue="model", ax=ax, linewidth=2,
                     style="model", dashes=REST_VS_EPOCH_STYLE["dashes"], palette=REST_VS_EPOCH_STYLE["palette"])
        ax.set_xlabel("Epoch number")
        ax.set_title(f"Subject {sid}")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Rest length vs epoch ({title_suffix})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / out_fname}")


def plot_rest_vs_epoch_nosim_and_1xsim():
    """Rest vs epoch for no-sim and 1x-sim; 
    skips if split file is missing."""
    for models_dir, split_path, out_fname, title_suffix in REST_VS_EPOCH_CONFIGS:
        if not Path(split_path).exists():
            print(f"Missing {split_path}; skipping {out_fname}")
            continue
        rest_vs_epoch_one_run(models_dir, str(split_path), out_fname, title_suffix)


def main():
    plot_no_sim_vs_1x_sim()
    plot_rest_vs_epoch_nosim_and_1xsim()
    print(f"Outputs in {OUT_DIR}")

if __name__ == "__main__":
    main()