"""
=============================================================
Transfer Learning: Model3 (ThesisDNN + XGBoost)
Comparison: Dataset1(orig) as Source vs Dataset2(clean) as Source

Generates a 3x2 comparison figure:
  Rows    : Transmission Gain / MCS / Airtime
  Columns : Direction-A (Orig→Clean)  |  Direction-B (Clean→Orig)
  Curves  : Mean RE (solid) / Max RE (dashed) / Min RE (dotted)

Usage (in Claude Code):
  cd /path/to/data
  python transfer_learning_model3.py

Requirements: pandas, numpy, torch, xgboost, scikit-learn, matplotlib
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# Paths – change if needed
# ─────────────────────────────────────────────────────────────
SOURCE_ORIG  =  "C:\BAI_project\origin_data\original_preprocess.csv"     # Dataset 1
SOURCE_CLEAN = "clean_ul_with_conditions2.csv" # Dataset 2
OUTPUT_FIG   = "transfer_learning_comparison.png"

import os, random, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# 0)  Global Config
# ─────────────────────────────────────────────────────────────
TARGET_COL     = "pm_power"
SEED           = 42
MIN_SLICE_SIZE = 20

# DNN Hyper-params  (matches thesis)
DNN_EPOCHS     = 400
DNN_PATIENCE   = 40
DNN_LR         = 1e-3
DNN_BATCH      = 32
DNN_WD         = 0.01

# Fine-tune hyper-params (transfer step – lighter than source training)
FT_EPOCHS      = 150
FT_PATIENCE    = 25
FT_LR          = 2e-4   # lower LR for fine-tuning
FT_BATCH       = 32

# XGBoost Hyper-params  (matches thesis)
XGB_PARAMS = dict(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    eta              = 0.22,
    max_depth        = 5,
    subsample        = 0.85,
    colsample_bytree = 0.9,
    min_child_weight = 3,
    reg_lambda       = 1.0,
    reg_alpha        = 0.1,
    seed             = SEED,
    tree_method      = "hist",
    device           = "cuda" if torch.cuda.is_available() else "cpu",
)
XGB_ROUNDS = 256
XGB_EARLY  = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] device={DEVICE} | seed={SEED}")

# ─────────────────────────────────────────────────────────────
# 1)  Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()

# ─────────────────────────────────────────────────────────────
# 2)  Column Mapping  (orig UL → clean naming)
#
#  RATIONALE:
#    original_preprocess has _ul / _dl suffix columns.
#    clean_ul_with_conditions2 uses clean names (UL-only).
#    We map the UL columns from orig so both DataFrames share
#    the same feature namespace for the transfer model.
# ─────────────────────────────────────────────────────────────
ORIG_TO_CLEAN = {
    "txgain_ul":       "txgain",
    "selected_mcs_ul": "selected_mcs",
    "airtime_ul":      "airtime",
    "nRBs_ul":         "nRBs",
    "mean_snr_ul":     "mean_snr",
    "bler_ul":         "bler",
    "thr_ul":          "thr",
    "bsr_ul":          "bsr",
    # shared columns kept as-is
    # turbodec_it, dec_time, num_ues, pm_power already match
}

# ─────────────────────────────────────────────────────────────
# 3)  Feature Engineering  (identical to thesis Model3)
# ─────────────────────────────────────────────────────────────
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    def has(*cols): return all(c in d.columns for c in cols)
    if has("txgain", "airtime"):
        d["txgain_x_airtime"] = d["txgain"] * d["airtime"]
    if has("selected_mcs", "airtime"):
        d["mcs_x_airtime"]    = d["selected_mcs"] * d["airtime"]
    if has("mean_snr", "bler"):
        d["snr_per_bler"]     = d["mean_snr"].astype(float) / (d["bler"].astype(float) + 1e-6)
    if has("thr", "airtime"):
        d["thr_per_airtime"]  = d["thr"].astype(float) / (d["airtime"].astype(float).clip(lower=0.01) + 1e-6)
    # clean-only engineered features are skipped for orig (no rssi/overflows)
    return d

# ─────────────────────────────────────────────────────────────
# 4)  Load Datasets
# ─────────────────────────────────────────────────────────────
# Features that exist in BOTH datasets after renaming
COMMON_BASE = ["txgain", "selected_mcs", "airtime", "nRBs",
               "mean_snr", "bler", "thr", "bsr",
               "turbodec_it", "dec_time", "num_ues"]
COMMON_ENG  = ["txgain_x_airtime", "mcs_x_airtime", "snr_per_bler", "thr_per_airtime"]
ALL_FEATS   = COMMON_BASE + COMMON_ENG   # 15 features used for transfer model

EXPERIMENTS = {
    "Transmission Gain": {"slice_col": "txgain",       "base_feats": ["selected_mcs", "airtime", "nRBs"]},
    "MCS":               {"slice_col": "selected_mcs", "base_feats": ["txgain",        "airtime", "nRBs"]},
    "Airtime":           {"slice_col": "airtime",      "base_feats": ["txgain", "selected_mcs", "nRBs"]},
}


def load_orig(path=SOURCE_ORIG):
    df = pd.read_csv(path)
    df = df.rename(columns=ORIG_TO_CLEAN)
    df = add_feature_engineering(df)
    df = df[df[TARGET_COL] > 0].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  [Dataset1/orig]  loaded {len(df):,} rows")
    return df


def load_clean(path=SOURCE_CLEAN):
    df = pd.read_csv(path)
    df = add_feature_engineering(df)
    df = df[df[TARGET_COL] > 0].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  [Dataset2/clean] loaded {len(df):,} rows")
    return df


def get_feature_cols(exp_cfg, df):
    """Return the feature columns that actually exist in df."""
    base   = [c for c in exp_cfg["base_feats"] + COMMON_BASE if c in df.columns]
    eng    = [c for c in COMMON_ENG if c in df.columns]
    feats  = list(dict.fromkeys(base + eng))   # deduplicate, preserve order
    return feats

# ─────────────────────────────────────────────────────────────
# 5)  Preprocessing helpers
# ─────────────────────────────────────────────────────────────
def clean_numeric(df, cols):
    d = df.dropna(subset=cols).copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=cols).copy()


def winsorize_fit(X, lo=1, hi=99):
    X_w, bounds = X.copy().astype(float), []
    for j in range(X.shape[1]):
        l, h = np.percentile(X_w[:, j], lo), np.percentile(X_w[:, j], hi)
        if h > l: X_w[:, j] = np.clip(X_w[:, j], l, h)
        bounds.append((l, h))
    return X_w, bounds


def winsorize_apply(X, bounds):
    X_w = X.copy().astype(float)
    for j, (l, h) in enumerate(bounds):
        if h > l: X_w[:, j] = np.clip(X_w[:, j], l, h)
    return X_w


def split_train_val_test(df, seed=SEED, train_r=0.8, val_r=0.1):
    tr, te = train_test_split(df, test_size=1-train_r, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_r,     random_state=seed)
    return tr, va, te

# ─────────────────────────────────────────────────────────────
# 6)  ThesisDNN  (exact replica from thesis Model3)
#     587 → 261 → 186 → 99 → bottleneck(16) → head(1)
#     No Dropout / BN, L2 via weight_decay
# ─────────────────────────────────────────────────────────────
class TabularDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
    def __len__(self):           return len(self.X)
    def __getitem__(self, i):    return self.X[i], self.y[i]


class ThesisDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1        = nn.Linear(input_dim, 587)
        self.fc2        = nn.Linear(587, 261)
        self.fc3        = nn.Linear(261, 186)
        self.fc4        = nn.Linear(186, 99)
        self.bottleneck = nn.Linear(99,  16)
        self.head       = nn.Linear(16,   1)

    def forward(self, x):
        h   = F.relu(self.fc1(x))
        h   = F.relu(self.fc2(h))
        h   = F.relu(self.fc3(h))
        h   = F.relu(self.fc4(h))
        emb = self.bottleneck(h)           # 16-dim (no activation)
        out = self.head(F.relu(emb))
        return out, emb

# ─────────────────────────────────────────────────────────────
# 7)  DNN Training (generic – used for both source & fine-tune)
# ─────────────────────────────────────────────────────────────
def train_dnn_generic(X_tr, y_tr, X_va, y_va,
                      input_dim, model=None,
                      epochs=DNN_EPOCHS, batch=DNN_BATCH,
                      lr=DNN_LR, wd=DNN_WD, patience=DNN_PATIENCE,
                      verbose_every=100, seed=SEED):
    set_seed(seed)
    if model is None:
        model = ThesisDNN(input_dim).to(DEVICE)
    else:
        model = model.to(DEVICE)

    loader    = DataLoader(TabularDS(X_tr, y_tr), batch_size=batch, shuffle=True)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).view(-1, 1).to(DEVICE)

    best_val, best_state, no_impr = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss_fn(pred, yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp, _ = model(X_va_t)
            vloss = loss_fn(vp, y_va_t).item()

        if vloss < best_val - 1e-8:
            best_val, best_state, no_impr = vloss, {k: v.cpu().clone()
                for k, v in model.state_dict().items()}, 0
        else:
            no_impr += 1

        if verbose_every and ep % verbose_every == 0:
            print(f"    ep {ep:04d}  val_MSE={vloss:.5f}  no_impr={no_impr}")

        if no_impr >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, X_s, batch_size=1024):
    model.eval()
    X_t  = torch.tensor(X_s, dtype=torch.float32)
    embs = []
    for i in range(0, len(X_t), batch_size):
        _, emb = model(X_t[i:i+batch_size].to(DEVICE))
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)

# ─────────────────────────────────────────────────────────────
# 8)  XGBoost Training
# ─────────────────────────────────────────────────────────────
def train_xgb(emb_tr, y_tr, emb_va, y_va,
              xgb_model=None, rounds=XGB_ROUNDS, seed=SEED):
    params = {**XGB_PARAMS, "seed": seed}
    dtrain = xgb.DMatrix(emb_tr, label=y_tr)
    dval   = xgb.DMatrix(emb_va, label=y_va)
    booster = xgb.train(
        params, dtrain,
        num_boost_round       = rounds,
        evals                 = [(dval, "val")],
        early_stopping_rounds = XGB_EARLY,
        verbose_eval          = False,
        xgb_model             = xgb_model,   # None=scratch, or source booster for warm-start
    )
    return booster

# ─────────────────────────────────────────────────────────────
# 9)  Metrics
# ─────────────────────────────────────────────────────────────
def relative_errors(y_true, y_pred, eps=1e-3):
    """Return point-wise RE (%) array."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return np.abs(yt - yp) / (np.abs(yt) + eps) * 100.0


def slice_re_stats(df_eval, y_pred, slice_col):
    """
    For each unique slice value, compute (mean_re, max_re, min_re).
    Returns a DataFrame with columns: slice_val, mean_re, max_re, min_re.
    """
    y_true = df_eval[TARGET_COL].values
    re_arr = relative_errors(y_true, y_pred)
    df_tmp = df_eval[[slice_col]].copy().reset_index(drop=True)
    df_tmp["re"] = re_arr
    agg = (df_tmp.groupby(slice_col)["re"]
                 .agg(mean_re="mean", max_re="max", min_re="min")
                 .reset_index()
                 .rename(columns={slice_col: "slice_val"}))
    return agg

# ─────────────────────────────────────────────────────────────
# 10)  Full Transfer Pipeline (SOURCE → TARGET)
#
#  Steps:
#   1. Pre-process source (all rows combined)
#   2. Train DNN on source
#   3. Fine-tune DNN on target (lower LR, fewer epochs)
#   4. Extract 16-dim embeddings from fine-tuned DNN
#   5. Warm-start XGBoost on target embeddings
#   6. Evaluate per-slice on target test set
# ─────────────────────────────────────────────────────────────
def run_transfer(df_src, df_tgt, exp_name, exp_cfg):
    print(f"\n  ── {exp_name} ──")
    slice_col   = exp_cfg["slice_col"]

    # ── feature columns that exist in BOTH datasets ──────────
    feat_src = get_feature_cols(exp_cfg, df_src)
    feat_tgt = get_feature_cols(exp_cfg, df_tgt)
    feats    = [f for f in feat_src if f in feat_tgt]
    print(f"    feats ({len(feats)}): {feats}")

    cols_src = feats + [slice_col, TARGET_COL]
    cols_tgt = feats + [slice_col, TARGET_COL]

    d_src = clean_numeric(df_src, [c for c in cols_src if c in df_src.columns])
    d_tgt = clean_numeric(df_tgt, [c for c in cols_tgt if c in df_tgt.columns])

    # ── SOURCE: 80/10/10 split ────────────────────────────────
    s_tr, s_va, s_te = split_train_val_test(d_src)

    X_s_tr = s_tr[feats].values.astype(float)
    y_s_tr = s_tr[TARGET_COL].values.astype(float)
    X_s_va = s_va[feats].values.astype(float)
    y_s_va = s_va[TARGET_COL].values.astype(float)

    X_s_tr_w, bounds_s = winsorize_fit(X_s_tr)
    X_s_va_w           = winsorize_apply(X_s_va, bounds_s)

    sc_src             = StandardScaler()
    X_s_tr_s           = sc_src.fit_transform(X_s_tr_w)
    X_s_va_s           = sc_src.transform(X_s_va_w)

    # ── Step 1: Train DNN on SOURCE ───────────────────────────
    print(f"    [Source DNN] training on {len(s_tr)+len(s_va)} samples …")
    dnn_src = train_dnn_generic(
        X_s_tr_s, y_s_tr, X_s_va_s, y_s_va,
        input_dim=len(feats), epochs=DNN_EPOCHS, batch=DNN_BATCH,
        lr=DNN_LR, wd=DNN_WD, patience=DNN_PATIENCE, seed=SEED)

    # ── Step 2: Extract embeddings & train XGBoost on SOURCE ─
    emb_s_tr = extract_embeddings(dnn_src, X_s_tr_s)
    emb_s_va = extract_embeddings(dnn_src, X_s_va_s)
    xgb_src  = train_xgb(emb_s_tr, y_s_tr, emb_s_va, y_s_va)

    # ── TARGET: 80/10/10 split ────────────────────────────────
    t_tr, t_va, t_te = split_train_val_test(d_tgt)

    X_t_tr = t_tr[feats].values.astype(float)
    y_t_tr = t_tr[TARGET_COL].values.astype(float)
    X_t_va = t_va[feats].values.astype(float)
    y_t_va = t_va[TARGET_COL].values.astype(float)
    X_t_te = t_te[feats].values.astype(float)

    X_t_tr_w, bounds_t = winsorize_fit(X_t_tr)
    X_t_va_w           = winsorize_apply(X_t_va, bounds_t)
    X_t_te_w           = winsorize_apply(X_t_te, bounds_t)

    sc_tgt             = StandardScaler()
    X_t_tr_s           = sc_tgt.fit_transform(X_t_tr_w)
    X_t_va_s           = sc_tgt.transform(X_t_va_w)
    X_t_te_s           = sc_tgt.transform(X_t_te_w)

    # ── Step 3: Fine-tune DNN on TARGET (transfer weights) ────
    #    Copy source DNN weights; unfreeze all layers; use smaller LR.
    import copy
    dnn_ft = copy.deepcopy(dnn_src)
    print(f"    [Fine-tune DNN] adapting on {len(t_tr)+len(t_va)} target samples …")
    dnn_ft = train_dnn_generic(
        X_t_tr_s, y_t_tr, X_t_va_s, y_t_va,
        input_dim=len(feats), model=dnn_ft,
        epochs=FT_EPOCHS, batch=FT_BATCH,
        lr=FT_LR, wd=DNN_WD, patience=FT_PATIENCE, seed=SEED)

    # ── Step 4: Extract embeddings from fine-tuned DNN ────────
    emb_t_tr = extract_embeddings(dnn_ft, X_t_tr_s)
    emb_t_va = extract_embeddings(dnn_ft, X_t_va_s)
    emb_t_te = extract_embeddings(dnn_ft, X_t_te_s)

    # ── Step 5: Warm-start XGBoost on TARGET embeddings ───────
    #    Starts from source booster, adds more trees for target.
    print(f"    [Warm-start XGB] adapting booster on target embeddings …")
    xgb_ft = train_xgb(emb_t_tr, y_t_tr, emb_t_va, y_t_va,
                        xgb_model=xgb_src, rounds=128)   # fewer extra rounds

    # ── Step 6: Predict & compute per-slice RE stats ──────────
    y_pred = xgb_ft.predict(xgb.DMatrix(emb_t_te))
    y_pred = np.clip(y_pred, y_t_tr.min() * 0.9, y_t_tr.max() * 1.1)

    df_eval = t_te.copy().reset_index(drop=True)
    stats   = slice_re_stats(df_eval, y_pred, slice_col)

    # overall metric
    re_all  = relative_errors(df_eval[TARGET_COL].values, y_pred)
    print(f"    Overall MRE = {re_all.mean():.2f}%  "
          f"(max={re_all.max():.1f}%  min={re_all.min():.1f}%)")

    return stats   # DataFrame: slice_val, mean_re, max_re, min_re

# ─────────────────────────────────────────────────────────────
# 11)  Main: Run both directions, collect results
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Loading datasets …")
print("="*60)
df_orig  = load_orig()
df_clean = load_clean()

# results[exp_name] = { "A": stats_df, "B": stats_df }
results = {}

for exp_name, exp_cfg in EXPERIMENTS.items():
    results[exp_name] = {}

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Direction A: Dataset1(orig) → Dataset2(clean)
    print("\n[Direction A]  Source=Dataset1(orig)  →  Target=Dataset2(clean)")
    results[exp_name]["A"] = run_transfer(df_orig, df_clean, exp_name, exp_cfg)

    # Direction B: Dataset2(clean) → Dataset1(orig)
    print("\n[Direction B]  Source=Dataset2(clean)  →  Target=Dataset1(orig)")
    results[exp_name]["B"] = run_transfer(df_clean, df_orig, exp_name, exp_cfg)

# ─────────────────────────────────────────────────────────────
# 12)  Plotting
# ─────────────────────────────────────────────────────────────
EXP_LABELS = {
    "Transmission Gain": "Transmission Gain (dBm)",
    "MCS":               "MCS Index",
    "Airtime":           "Airtime Ratio",
}

DIR_LABELS = {
    "A": "Source = Dataset1 (original_preprocess)\nTarget = Dataset2 (clean_ul)",
    "B": "Source = Dataset2 (clean_ul)\nTarget = Dataset1 (original_preprocess)",
}

# Color scheme
COL_MEAN = "#1f77b4"   # blue
COL_MAX  = "#d62728"   # red
COL_MIN  = "#2ca02c"   # green

fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(16, 14),
    constrained_layout=True
)
fig.suptitle(
    "Transfer Learning Comparison: Model3 (ThesisDNN + XGBoost)\n"
    "Mean / Max / Min Relative Error (%) across Slices",
    fontsize=15, fontweight="bold", y=1.01
)

for row_idx, (exp_name, exp_cfg) in enumerate(EXPERIMENTS.items()):
    for col_idx, direction in enumerate(["A", "B"]):
        ax  = axes[row_idx, col_idx]
        st  = results[exp_name][direction]

        xv  = st["slice_val"].values
        mn  = st["mean_re"].values
        mx  = st["max_re"].values
        mi  = st["min_re"].values

        # Fill band between min and max
        ax.fill_between(xv, mi, mx,
                        alpha=0.12, color=COL_MEAN, label="_nolegend_")

        ax.plot(xv, mn, color=COL_MEAN, lw=2.5,
                marker="o", ms=5, label="Mean RE")
        ax.plot(xv, mx, color=COL_MAX,  lw=1.8,
                ls="--", marker="^", ms=4, label="Max RE")
        ax.plot(xv, mi, color=COL_MIN,  lw=1.8,
                ls=":",  marker="v", ms=4, label="Min RE")

        # Annotate overall mean
        ax.axhline(mn.mean(), color=COL_MEAN, lw=1.0, ls="-.", alpha=0.6)
        ax.text(xv[-1], mn.mean() * 1.02,
                f"avg={mn.mean():.1f}%",
                fontsize=8, color=COL_MEAN, ha="right")

        ax.set_xlabel(EXP_LABELS[exp_name], fontsize=9)
        ax.set_ylabel("Relative Error (%)", fontsize=9)
        ax.set_title(
            f"{'A' if col_idx==0 else 'B'})  {exp_name}\n"
            f"{DIR_LABELS[direction]}",
            fontsize=8.5, loc="left"
        )
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

# Column headers
for col_idx, direction in enumerate(["A", "B"]):
    axes[0, col_idx].set_title(
        f"{'Direction A' if col_idx==0 else 'Direction B'}:  "
        + ("orig → clean" if col_idx==0 else "clean → orig")
        + f"\n{EXPERIMENTS[list(EXPERIMENTS.keys())[0]]['slice_col']}  |  "
        + f"{'Mean' if col_idx==0 else 'Max / Min'} RE (%) curve",
        fontsize=9, pad=6
    )

plt.savefig(OUTPUT_FIG, dpi=180, bbox_inches="tight")
print(f"\n[Done]  Figure saved → {OUTPUT_FIG}")

# ─────────────────────────────────────────────────────────────
# 13)  Print summary table
# ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: Mean RE (%) per experiment and direction")
print("="*70)
print(f"{'Experiment':<22} {'Direction':>10} {'Mean_RE':>10} {'Max_RE':>10} {'Min_RE':>10}")
print("-"*70)
for exp_name in EXPERIMENTS:
    for direction in ["A", "B"]:
        st = results[exp_name][direction]
        print(f"{exp_name:<22} {direction:>10} "
              f"{st['mean_re'].mean():>10.2f} "
              f"{st['max_re'].max():>10.2f} "
              f"{st['min_re'].min():>10.2f}")
print("="*70)
