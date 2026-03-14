"""
Train LSTM to predict rest_length from trial history. 
Splits: dataset (train=original, test=replication) or pooled (80/20).
Uses 20% of the training set as LSTM val

copy command: 
uv run python scripts/train_lstm.py [--split pooled --load-split splits/pooled_80_20]

THIS IS USUALLY ONLY CALLED FROM RUN_ALL_MODELS.py in my pipeline
"""
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import (
    MAIN_RESPONSE_TRIAL_TYPES,
    get_trials,
    get_epoch_table,
    add_trial_features,
    get_trial_feature_columns,
    build_trial_sequences,
)
from src.split import split_by_dataset, train_test_split_pooled, load_split_ids
from src.lstm_model import RestLSTM

###############
# helper functions:
def to_tensors(X, y, lengths):
    """from numpy to float32/long torch"""
    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(lengths).long()


def fit_scaler_and_transform(X_train, X_val, X_test, len_train):
    """
    len_train = how much actual data there is without padding
    right now in my setup there isn't padding, but only include 
    the correct length just in case.
    Scale relevant features for the LSTM training
    """
    n_subj, _, n_feat = X_train.shape # n subjects, seq len, num features

    #scale:
    un_padded_data = np.concatenate([X_train[i, :len_train[i], :] for i in range(n_subj)], axis=0)
    scaler = StandardScaler().fit(un_padded_data)

    # scale them all:
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape).astype(np.float32)
    
    return scaler, X_train, X_val, X_test

def train_epoch(model, X, y, lengths, optimizer, batch_size):
    """One epoch over batches; returns mean MSE on valid target positions."""
    model.train()
    n_subj = len(X)
    idx = torch.randperm(n_subj)

    # sums up total loss only for predictions after entire task epochs
    # counts number of predictions after entire task epoch (should be 30 per subj)
    total_loss, n_valid_targets = 0.0, 0

    # iterate through the batches
    for start in range(0, n_subj, batch_size):
        batch_index = idx[start : start + batch_size] # selects subjects
        xbatch, ybatch = X[batch_index], y[batch_index] # gets those subjects
        lb = lengths[batch_index] # gets num valid trials for each subject 

        # run the model on this batch of subjects output: batch size x sequence length
        out = model(xbatch, lengths=lb).squeeze(-1)
        y_trunc = ybatch[:,:out.shape[1]]

        # valid targets are only at epoch-end positions (y is NaN everywhere else)
        mask = ~torch.isnan(y_trunc)
        n_t = mask.sum().item()
        if n_t == 0:
            continue

        loss = ((out[mask] - y_trunc[mask]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * n_t
        n_valid_targets += n_t

    return total_loss / n_valid_targets if n_valid_targets > 0 else 0.0


@torch.no_grad() # makes sure it doesn't store tensors for backward pass since this is just eval
def evaluate(model, X, y, lengths, batch_size=32):
    """Return MAE and R² at the valid (end of epoch) positions."""
    model.eval()
    preds, actuals = [], []

    for start in range(0, len(X), batch_size): #doing batch size in case of mmeory issues
        batch_index = slice(start, min(start + batch_size, len(X)))
        out = model(X[batch_index], lengths=lengths[batch_index]).squeeze(-1).cpu()
        ybatch = y[batch_index]
        for i, L in enumerate(lengths[batch_index]):
            L = int(L)
            v = ~torch.isnan(ybatch[i, :L])
            preds.extend(out[i, :L][v].tolist())
            actuals.extend(ybatch[i, :L][v].tolist())

    return mean_absolute_error(actuals, np.array(preds)), r2_score(actuals, np.array(preds))


def split_train_val_subjects(train_ids, epoch_table, val_frac=0.2, seed=42):
    """Split the train subjects into train/val for LSTM purposes
    val_frac = proportion of training set to separate for LSTM validation during training
    use pooled split when required
    all simulated subjects (Simulated_*) will stay in train
    returns train subject ids, val subject ids"""

    rng = np.random.RandomState(seed)
    real = sorted(s for s in train_ids if not s.startswith("simulated_"))
    sim = train_ids - set(real)
    val_ids = []

    # if the data is pooled (instead of train = original, test = replication)
    if "dataset" in epoch_table.columns:
        ds = epoch_table.drop_duplicates("subject_id").set_index("subject_id")["dataset"]
        known = [s for s in real if s in ds.index]
        for dname in ds.loc[known].unique():
            sids = [s for s in known if ds.get(s) == dname]
            rng.shuffle(sids)
            val_ids.extend(sids[: max(1, int(len(sids) * val_frac))])
    else:
        rng.shuffle(real)
        val_ids = real[: max(1, int(len(real) * val_frac))]
    
    train_ids = (set(real) - set(val_ids)) | sim # adds the sim subjects to train
    val_ids = set(val_ids)
    return train_ids, val_ids

#####################
# RUN ACTUAL MODEL:

def train_and_validate(X_train, y_train, len_train, X_val, y_val, len_val, n_feat, hidden, layers, dropout, lr, wd, batch_size, epochs):
    """Train LSTM with early stopping (patience 15). Returns best_val_mae, best_state_dict."""
    model = RestLSTM(n_features=n_feat, hidden_size=hidden, num_layers=layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_mae, best_state, pat = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        train_epoch(model, X_train, y_train, len_train, opt, batch_size)
        v_mae, v_r2 = evaluate(model, X_val, y_val, len_val)
        if v_mae < best_mae:
            best_mae, best_state, pat = v_mae, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep}: val_MAE={v_mae:.4f} val_R²={v_r2:.4f}")
        if pat >= 15:
            print(f"  Early stop at epoch {ep}")
            break
    return best_mae, best_state

def run_lstm(split="dataset", load_split=None, test_frac=0.2, val_frac=0.2, seed=42, epochs=100, hidden=64, layers=2, dropout=0.3, lr=2e-3, weight_decay=1e-3, batch_size=8, save=None, trials_df=None, epoch_table_df=None):
    """
    Train the LSTM on trial-level data; 
    test frac here only matters if calling this function from train_lstm.py instead of
    run_all_models.py (it uses it to make its own split if a split is not already loaded)
    return dict of split, lstm_mae, lstm_r2, feature_cols, model params."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # get the data:
    if trials_df is not None and epoch_table_df is not None:
        trials = trials_df.copy()
        epoch_table = epoch_table_df.copy()
    else:
        epoch_table = get_epoch_table()
        trials = get_trials()
        trials = trials[trials["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)].copy()
        trials = add_trial_features(trials, epoch_table)
    feature_cols = get_trial_feature_columns(include_dataset=(split == "pooled"))
    trials = trials.dropna(subset=feature_cols)

    # do the split by dataset (train = original)
    # in this case there is no sim
    if split == "dataset":
        train_epochs, test_epochs = split_by_dataset(epoch_table, "original", "replication")
        train_ids = set(train_epochs["subject_id"].unique())
        test_ids = set(test_epochs["subject_id"].unique())
    else: # the pooled split
        if load_split:
            tids, teids = load_split_ids(load_split)
            train_ids, test_ids = set(tids), set(teids)
        else:
            train_epochs, test_epochs = train_test_split_pooled(epoch_table, test_frac=test_frac, random_state=seed)
            train_ids = set(train_epochs["subject_id"].unique())
            test_ids = set(test_epochs["subject_id"].unique())
    train_ids, val_ids = split_train_val_subjects(train_ids, epoch_table, val_frac=val_frac, seed=seed)
    
    print(f"LSTM has: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test subjects")

    train_trials = trials[trials["subject_id"].isin(train_ids)]
    val_trials = trials[trials["subject_id"].isin(val_ids)]
    test_trials = trials[trials["subject_id"].isin(test_ids)]
    X_train, y_train, len_train = build_trial_sequences(train_trials, epoch_table, feature_cols)
    X_val, y_val, len_val = build_trial_sequences(val_trials, epoch_table, feature_cols)
    X_test, y_test, len_test = build_trial_sequences(test_trials, epoch_table, feature_cols)

    # scale the data
    scaler, X_train, X_val, X_test = fit_scaler_and_transform(X_train, X_val, X_test, len_train)
    n_feat = X_train.shape[2]
    X_train, y_train, len_train = to_tensors(X_train, y_train, len_train)
    X_val, y_val, len_val = to_tensors(X_val, y_val, len_val)
    X_test, y_test, len_test = to_tensors(X_test, y_test, len_test)

    # get best weights from training
    best_val_mae, best_state = train_and_validate(X_train, y_train, len_train, X_val, y_val, len_val, n_feat, hidden, layers, dropout, lr, weight_decay, batch_size, epochs)
    # load best weights into the model:
    model = RestLSTM(n_features=n_feat, hidden_size=hidden, num_layers=layers, dropout=dropout)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save and best_state is not None: # save is not none when called frmo run_all_models.py
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        d = {"model_state_dict": best_state, "hidden": hidden, "layers": layers, "dropout": dropout}
        d["scaler_mean"], d["scaler_scale"] = scaler.mean_, scaler.scale_
        torch.save(d, save)
        Path(str(save).replace(".pt", "_features.txt")).write_text("\n".join(feature_cols))

    mae, r2 = evaluate(model, X_test, y_test, len_test)
    print(f"  Best val MAE: {best_val_mae:.4f} → Test MAE: {mae:.4f}, R²: {r2:.4f}")
    return {"split": split, "lstm_mae": float(mae), "lstm_r2": float(r2), "feature_cols": feature_cols, "params": {"epochs": epochs, "hidden": hidden, "layers": layers, "dropout": dropout, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "val_frac": val_frac}}

def main():
    p = argparse.ArgumentParser(description="Train LSTM")
    p.add_argument("--split", choices=["dataset", "pooled"], default="dataset")
    p.add_argument("--load-split", type=str, default=None)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()
    res = run_lstm(**vars(args))
    print(f"\nFinal Test MAE: {res['lstm_mae']:.4f}, R²: {res['lstm_r2']:.4f}")

if __name__ == "__main__":
    main()