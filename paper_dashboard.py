#!/usr/bin/env python3
"""
Dashboard v5.1 — EvoX-Boost (Final)
Generates all publication-ready figures:
 - SHAP bar + beeswarm (CPU-cloned EvoX-Boost)
 - Confusion matrices (RF, XGB)
 - TP/TN/FP/FN square diagrams
 - Micro & macro metrics CSV
 - XGB ROC (prob_* columns)
 - Accuracy comparison
 - Runtime plot
 - Model size
 - Correlation heatmap
Saves outputs in ./output/figs/
"""

import os, json, glob, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Attempt to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ======================================================================
# Folders & settings
# ======================================================================
OUT = "./output"
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

sns.set_style("white")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10
})

# ======================================================================
# Utility functions
# ======================================================================

def save_fig(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)

def read_labels():
    path = os.path.join(OUT, "decoded_labels.json")
    if not os.path.exists(path):
        raise FileNotFoundError("decoded_labels.json missing in ./output")
    with open(path) as f:
        jm = json.load(f)
    return [jm[str(i)] for i in range(len(jm))]

def collect_preds(prefix):
    files = sorted(glob.glob(os.path.join(OUT, f"{prefix}_pred_fold_*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)

# ======================================================================
# 1) Correlation Heatmap
# ======================================================================

def plot_corr():
    if not os.path.exists("thyroid_cleaned.csv"):
        print("[corr] thyroid_cleaned.csv missing")
        return

    df = pd.read_csv("thyroid_cleaned.csv")
    df = df.drop(columns=["class"], errors="ignore")
    df = pd.get_dummies(df)

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)  # crisp & readable

    sns.heatmap(
        corr,
        cmap="Blues",         # 💙 clean blue theme
        annot=True,           # show all values
        fmt=".2f",            # format numbers
        annot_kws={"size": 5},  # small readable font
        linewidths=0.4,
        linecolor="black",
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    plt.xticks(rotation=60, fontsize=6)
    plt.yticks(fontsize=6)
    ax.set_title("Correlation Heatmap (Thyroid Dataset)", fontsize=12)

    save_fig(fig, "fig_3_corr_heatmap.png")
    print("[corr] saved")

# ======================================================================
# 2) Feature Importance (XGBoost Gain + Permutation Importance)
# ======================================================================

from sklearn.inspection import permutation_importance

def plot_feature_importance(labels, topk=25):

    print("[feat] Computing Gain and Permutation Importance for EvoX-Boost…")

    # Load clean data
    df = pd.read_csv("thyroid_cleaned.csv")
    Xraw = df.drop(columns=["class"])
    X = pd.get_dummies(Xraw)                    # same preprocessing used in training
    y_enc = pd.factorize(df["class"].astype(str))[0]

    # Load tuned params
    params = json.load(open("./output/xgb_tuned_params.json"))
    clean_params = {}
    for k, v in params.items():
        clean_params[k] = int(v) if k in ("max_depth","n_estimators") else float(v)

    # Train CPU clone of EvoX-Boost (no GPU needed here)
    clean_params["device"] = "cpu"
    clean_params["tree_method"] = "hist"
    clean_params["verbosity"] = 0

    model = XGBClassifier(**clean_params)
    model.fit(X, y_enc)

    # -----------------------------------------------------------
    # 1) XGBoost GAIN Importance (Most common in literature)
    # -----------------------------------------------------------
    gain_scores = model.get_booster().get_score(importance_type="gain")

    # Map features correctly
    imp_gain = pd.DataFrame({
        "feature": list(gain_scores.keys()),
        "gain": list(gain_scores.values())
    }).sort_values("gain", ascending=False).head(topk)

    # Plot Gain importance
    fig, ax = plt.subplots(figsize=(8, topk*0.28 + 1.5))
    sns.barplot(x="gain", y="feature", data=imp_gain, palette="deep", ax=ax)
    ax.set_title(f"XGBoost Gain Feature Importance (Top {topk}) - EvoX-Boost")

    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}",
                    (p.get_width()+1e-6, p.get_y()+p.get_height()/2),
                    fontsize=8)

    save_fig(fig, "fig_2_gain_importance.png")
    print("[feat] Gain importance saved")

    # -----------------------------------------------------------
    # 2) Permutation Importance (stable, model-agnostic)
    # -----------------------------------------------------------

    print("[feat] Running Permutation Importance (this may take ~20–90 sec)…")
    
    perm = permutation_importance(
        estimator=model,
        X=X,
        y=y_enc,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

    perm_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False).head(topk)

    # Plot Permutation Importance
    fig, ax = plt.subplots(figsize=(8, topk*0.28 + 1.5))
    sns.barplot(x="importance", y="feature", data=perm_imp, palette="muted", ax=ax)
    ax.set_title(f"Permutation Importance (Top {topk}) - EvoX-Boost")

    for p in ax.patches:
        ax.annotate(f"{p.get_width():.4f}",
                    (p.get_width()+1e-6, p.get_y()+p.get_height()/2),
                    fontsize=8)

    save_fig(fig, "fig_2_perm_importance.png")
    print("[feat] Permutation importance saved")

# ======================================================================
# 3) Confusion Matrix + TP/TN/FP/FN Square + Metrics
# ======================================================================

def plot_confusion_and_metrics(labels):
    models = [("RF","rf"), ("XGB","xgb")]
    rows = []

    for name, pref in models:
        df = collect_preds(pref)
        if df is None:
            print(f"[skip] No predictions for {name}")
            continue

        y_true = df["y_true"].astype(int).values
        y_pred = df["y_pred"].astype(int).values

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

        # ==== Metrics ====
        acc = accuracy_score(y_true, y_pred)
        prec_m = precision_score(y_true, y_pred, average="micro", zero_division=0)
        rec_m = recall_score(y_true, y_pred, average="micro", zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="micro", zero_division=0)
        prec_M = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_M = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_M = f1_score(y_true, y_pred, average="macro", zero_division=0)

        rows.append((name, acc, prec_m, rec_m, f1_m, prec_M, rec_M, f1_M))

        # ==== CONFUSION MATRIX (CLEAN GRID) ====
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw heatmap WITHOUT automatic grid lines
        sns.heatmap(
            cm,
            annot=False,
            fmt="",
            cmap=sns.light_palette("navy", as_cmap=True),
            cbar=True,
            square=True,
            linewidths=0.4,            # much thinner lines
            linecolor="black",         # clean sharp grid
            ax=ax
        )

        # Custom centered text (small, clean)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(
                j + 0.5, i + 0.5,
                str(val),
                ha="center", va="center",
                fontsize=7,
                color="black"
            )

        # Set ticks at cell centers
        ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
        ax.set_yticks(np.arange(cm.shape[0]) + 0.5)

        # Set labels properly (0..14)
        ax.set_xticklabels(range(cm.shape[1]))
        ax.set_yticklabels(range(cm.shape[0]))

        # Draw thin grid lines on cell boundaries
        ax.set_xticks(np.arange(cm.shape[1] + 1), minor=True)
        ax.set_yticks(np.arange(cm.shape[0] + 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.4)

        # Disable major grid
        ax.grid(False)


        ax.set_title(f"Confusion Matrix — {name}")
        save_fig(fig, f"{pref}_confusion_matrix.png")

        # ==== TP/TN/FP/FN square — Reference Style Version ====

        tp = int(np.trace(cm))
        fn = cm.sum(axis=1) - np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = int(cm.sum() - (tp + fn.sum() + fp.sum()))

        fp_sum = int(fp.sum())
        fn_sum = int(fn.sum())

        # Percentages
        TPP = tp / (tp + fn_sum) if (tp + fn_sum) > 0 else 0
        FNP = fn_sum / (tp + fn_sum) if (tp + fn_sum) > 0 else 0
        FPP = fp_sum / (fp_sum + tn) if (fp_sum + tn) > 0 else 0
        TNP = tn / (tn + fp_sum) if (tn + fp_sum) > 0 else 0

        # Colors based on your reference image
        COLOR_TP = "#356d8c"   # dark teal
        COLOR_FN = "#a8d4f0"   # light blue
        COLOR_FP = "#a8d4f0"   # light blue
        COLOR_TN = "#356d8c"   # dark teal

        colors = np.array([
            [COLOR_TP, COLOR_FN],    # row: actual positive
            [COLOR_FP, COLOR_TN]     # row: actual negative
        ])

        # Matrix in TP-FN / FP-TN layout
        sq = np.array([[tp, fn_sum], [fp_sum, tn]])

        fig, ax = plt.subplots(figsize=(5.8, 5.8))

        # Draw colored blocks (remove seaborn heatmap entirely)
        for i in range(2):
            for j in range(2):
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, color=colors[i][j])
                )

        # Auto text color based on background luminance
        def auto_color(hex_color):
            rgb = np.array([int(hex_color[k:k+2], 16)/255 for k in (1,3,5)])
            lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            return "white" if lum < 0.5 else "black"

        # Helper to annotate block values
        def annotate(x, y, label, count, pct, bg):
            col = auto_color(bg)

            # LABEL (TP, FP, etc.)
            ax.text(x+0.5, y+0.32, label,
                    ha="center", va="center",
                    fontsize=24, fontweight="bold",
                    color=col)

            # COUNT
            ax.text(x+0.5, y+0.60, f"{count}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color=col)

            # PERCENTAGE
            ax.text(x+0.5, y+0.82, f"({pct:.2%})",
                    ha="center", va="center",
                    fontsize=11, color=col)

        # TP, FN, FP, TN annotations
        annotate(0, 0, "TP", tp, TPP, COLOR_TP)
        annotate(1, 0, "FN", fn_sum, FNP, COLOR_FN)
        annotate(0, 1, "FP", fp_sum, FPP, COLOR_FP)
        annotate(1, 1, "TN", tn, TNP, COLOR_TN)

        # Axes formatting
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Predicted Positive", "Predicted Negative"], fontsize=11)
        ax.set_yticklabels(["Actual Positive", "Actual Negative"], fontsize=11)

        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_aspect("equal")

        ax.set_title(f"TP/TN/FP/FN — {name}", fontsize=16, pad=20)
        save_fig(fig, f"{pref}_tptnfpfn_square.png")

        # ==== Metrics Bar ====
        fig, ax = plt.subplots(figsize=(5,4))
        vals = [acc, prec_m, rec_m, f1_m]
        sns.barplot(
            x=["Accuracy","Precision","Recall","F1"],
            y=vals,
            hue=["Accuracy","Precision","Recall","F1"],
            dodge=False,
            palette="muted",
            legend=False,
            ax=ax
        )
        ax.set_ylim(0,1.05)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}",
                        (p.get_x()+p.get_width()/2., p.get_height()),
                        ha="center", va="bottom", fontsize=9)

        ax.set_title(f"{name} (Micro Metrics)")
        save_fig(fig, f"{pref}_metrics_micro.png")

    # Save metrics CSV
    dfm = pd.DataFrame(rows, columns=[
        "Model","Acc_micro","Prec_micro","Rec_micro","F1_micro",
        "Prec_macro","Rec_macro","F1_macro"
    ])
    dfm.to_csv(os.path.join(OUT, "final_metrics_summary.csv"), index=False)
    print("[metrics] saved final_metrics_summary.csv")

# ======================================================================
# 4) XGB ROC (RF lacks prob_* so no RF ROC)
# ======================================================================

def plot_roc_xgb(labels):
    df = collect_preds("xgb")
    if df is None:
        print("[roc] No XGB preds found")
        return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        print("[roc] No prob_* columns found in XGB files")
        return

    probs = df[prob_cols].values
    y_true = df["y_true"].astype(int).values
    Y = label_binarize(y_true, classes=list(range(len(labels))))

    # ROC curves
    fpr, tpr, roc_auc = {}, {}, {}

    # per-class
    for i in range(len(labels)):
        try:
            fpr[i], tpr[i], _ = roc_curve(Y[:,i], probs[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i] = np.array([0,1])
            tpr[i] = np.array([0,1])
            roc_auc[i] = 0.0

    # micro-average
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        fpr["micro"] = np.array([0,1])
        tpr["micro"] = np.array([0,1])
        roc_auc["micro"] = 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro (AUC={roc_auc['micro']:.3f})",
            color="black", linewidth=2)

    # top 4 classes by AUC
    top_classes = sorted(
        [(i, roc_auc[i]) for i in range(len(labels))],
        key=lambda x: x[1],
        reverse=True
    )[:4]

    colors = sns.color_palette("tab10", n_colors=6)
    for idx,(i,a) in enumerate(top_classes):
        ax.plot(fpr[i], tpr[i],
                label=f"{labels[i]} (AUC={a:.3f})",
                color=colors[idx], linewidth=1.5)

    ax.plot([0,1],[0,1],"k--",linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — XGB (Top Classes + Micro Avg)")
    ax.legend(fontsize=8, loc="lower right")

    save_fig(fig, "fig_7_roc_xgb.png")
    print("[roc] saved XGB ROC")

# ======================================================================
# 5) Accuracy Comparison
# ======================================================================

def plot_accuracy_comparison():
    path = os.path.join(OUT, "extreme_run_gpu_summary.csv")
    if not os.path.exists(path):
        print("[acc] summary missing")
        return

    sm = pd.read_csv(path)
    means = {}
    for col in ["RF","XGB"]:
        if col in sm.columns:
            means[col] = np.nanmean(pd.to_numeric(sm[col], errors="coerce"))

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(list(means.keys()), list(means.values()),
                  color=sns.color_palette("tab10")[:2])
    ax.set_ylim(0,1.0)
    for b in bars:
        ax.annotate(f"{b.get_height():.4f}",
                    (b.get_x()+b.get_width()/2., b.get_height()),
                    ha="center", va="bottom", fontsize=9)

    ax.set_title("Mean Accuracy (RF vs XGB)")
    save_fig(fig, "fig_6_accuracy_comparison.png")
    print("[acc] saved")

# ======================================================================
# 6) Runtime & Model Size
# ======================================================================

def plot_runtime_and_size():

    # ----- Runtime plot -----
    log_path = os.path.join(OUT, "fold_runtime_log.txt")
    times = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for ln in f:
                if "=" in ln:
                    try:
                        t = float(ln.strip().split("=")[1].replace("s",""))
                        times.append(t)
                    except:
                        pass

    if times:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1,len(times)+1), times, marker="o")
        for i,t in enumerate(times):
            ax.text(i+1, t+0.02, f"{t:.2f}s", ha="center", fontsize=9)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Fold Runtime (Seconds)")
        save_fig(fig, "fig_13_fold_runtime.png")
        print("[time] saved runtime")

    # ----- Model sizes -----
    files = []
    labels = []
    if os.path.exists(os.path.join(OUT, "xgb_model.pkl")):
        files.append(os.path.join(OUT, "xgb_model.pkl"))
        labels.append("XGB Model")

    if files:
        sizes = [os.path.getsize(f)/(1024*1024) for f in files]
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(labels, sizes, color=sns.color_palette("tab10")[:len(labels)])
        for b in bars:
            ax.annotate(f"{b.get_height():.2f} MB",
                        (b.get_x()+b.get_width()/2., b.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title("Model Size (MB)")
        save_fig(fig, "fig_model_sizes.png")
        print("[size] saved model sizes")

# ======================================================================
# MAIN
# ======================================================================

def main():
    warnings.filterwarnings("ignore")
    labels = read_labels()
    plot_corr()
    plot_feature_importance(labels, topk=25)
    plot_confusion_and_metrics(labels)
    plot_roc_xgb(labels)
    plot_accuracy_comparison()
    plot_runtime_and_size()
    print("\nAll figures saved to ./output/figs/")

if __name__ == "__main__":
    main()