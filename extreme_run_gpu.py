#!/usr/bin/env python3
# extreme_run_gpu.py — Final stable-flow GPU script (Optuna -> DE -> GA+DE -> 5-fold CV)
# Matches stable version logs/format, uses GPU XGBoost (tree_method="hist", device="cuda")
# LightGBM verbosity suppressed. GA+DE uses 3-fold in fitness (like stable).
import os, time, json, random, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# GPU/cuml imports (optional)
try:
    import cudf, cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_CUML = True
except Exception:
    cudf = None; cp = None; cuRF = None
    GPU_CUML = False

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import differential_evolution
import copy

# ---------------- CONFIG ----------------
DATA_PATH = "./thyroid_cleaned.csv"
TARGET_COL = "class"
SEED = 42
N_TRIALS = 25

random.seed(SEED); np.random.seed(SEED)
os.makedirs("./output", exist_ok=True)
os.makedirs("./output/figs", exist_ok=True)

def gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

GPU_ON = gpu_available()
print("GPU available:", GPU_ON)

# ---------------- LOAD & PREPROCESS ----------------
print("\n=== Loading data ===")
df = pd.read_csv(DATA_PATH)

  
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
cat_cols = [c for c in df.columns if c not in num_cols + [TARGET_COL]]

if num_cols:
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])

for c in cat_cols:
    df[c] = df[c].fillna("missing").astype(str)

if num_cols:
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

if cat_cols:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc_arr = enc.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

# label encode
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL].astype(str))
label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
json.dump(label_map, open("./output/decoded_labels.json", "w"), indent=4)
X = df.drop(columns=[TARGET_COL]).reset_index(drop=True)
# Save processed feature matrix for Objective-2
X.to_csv("./output/final_processed_X.csv", index=False)
pd.Series(y).to_csv("./output/final_labels.csv", index=False)

print("Saved final_processed_X.csv and final_labels.csv")
n_classes = len(le.classes_)

# adaptive folds (stable logic)
min_count = pd.Series(y).value_counts().min()
N_FOLDS = 7 if min_count >= 7 else 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("Classes:", le.classes_.tolist())
print("Adaptive CV folds:", N_FOLDS)

# ---------------- model builders ----------------
def make_rf(params=None):
    if GPU_CUML:
        base = dict(n_estimators=500, max_depth=20, random_state=SEED)
        if params: base.update(params)
        return cuRF(**base)
    else:
        from sklearn.ensemble import RandomForestClassifier as SKRF
        base = dict(n_estimators=500, max_depth=20, random_state=SEED, n_jobs=-1)
        if params: base.update(params)
        return SKRF(**base)

def make_xgb(params=None):
    base = dict(tree_method="hist", device="cuda" if GPU_ON else "cpu", random_state=SEED, verbosity=0)
    if params: base.update(params)
    return XGBClassifier(**base)

def make_lgb(params=None):
    # silence LightGBM info/warning printouts
    base = dict(n_estimators=500, random_state=SEED, verbosity=-1)
    if params: base.update(params)
    return lgb.LGBMClassifier(**base)

# ---------------- utilities ----------------
def safe_cast(params):
    out = {}
    for k, v in params.items():
        try:
            if k in ("n_estimators", "max_depth"):
                out[k] = int(round(float(v)))
            else:
                out[k] = float(v)
        except Exception:
            out[k] = v
    return out

def clamp(p):
    # ensure valid ranges for xgboost params
    p = dict(p)
    p["learning_rate"] = float(np.clip(p.get("learning_rate", 0.01), 0.001, 0.3))
    p["max_depth"] = int(np.clip(int(round(p.get("max_depth", 6))), 3, 24))
    p["subsample"] = float(np.clip(p.get("subsample", 0.9), 0.4, 1.0))
    p["colsample_bytree"] = float(np.clip(p.get("colsample_bytree", 0.9), 0.4, 1.0))
    p["n_estimators"] = int(np.clip(int(round(p.get("n_estimators", 200))), 50, 2000))
    p["min_child_weight"] = float(np.clip(p.get("min_child_weight", 1.0), 0.01, 100.0))
    p["gamma"] = float(np.clip(p.get("gamma", 0.0), 0.0, 5.0))
    p["reg_lambda"] = float(np.clip(p.get("reg_lambda", 0.0), 0.0, 10.0))
    return p

# ---------------- xgb cv scorer (robust) ----------------
def xgb_cv_score(params, Xdf, yarr, folds=3, seed=SEED):
    p = clamp(safe_cast(params))
    skf_local = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf_local.split(Xdf, yarr):
        Xtr = Xdf.iloc[tr].copy()
        Xte = Xdf.iloc[te].copy()
        ytr = yarr[tr].copy()
        yte = yarr[te].copy()

        missing = np.setdiff1d(np.arange(n_classes), np.unique(ytr))
        if len(missing):
            dummy = np.zeros((len(missing), Xtr.shape[1]))
            Xtr = np.vstack([Xtr.values, dummy])
            ytr = np.concatenate([ytr, missing])
        else:
            Xtr = Xtr.values

        try:
            m = make_xgb(p)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte.values)
            scores.append(accuracy_score(yte, pred))
        except Exception:
            # in case of failure inside CV fold, penalize strongly (0.0)
            scores.append(0.0)
    return float(np.mean(scores))

# ---------------- 1) OPTUNA ----------------
# --- Convergence logging ---
convergence_log = []
optuna_start = time.time()
print("\n=== Optuna search (XGBoost) ===")
def optuna_objective(trial):
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06),
        "max_depth": trial.suggest_int("max_depth", 6, 18),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 6.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
    }
    
    # compute score
    score = xgb_cv_score(p, X, y, folds=3, seed=SEED)

    # ---- Convergence logging ----
    convergence_log.append({
        "iteration": trial.number,
        "score": score
    })

    return score
    
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=True)
optuna_time = time.time() - optuna_start
optuna_best = study.best_params
optuna_score = study.best_value
pd.DataFrame(convergence_log).to_csv(
    "./output/convergence_history.csv",
    index=False
)
print(f"\nOptuna best mean acc= {optuna_score:.5f}")
print("Optuna best params:", optuna_best)

# ---------------- 2) DE refinement ----------------
print("\n=== Differential Evolution refinement ===")
de_start = time.time()
def de_bounds_from_opt(opt):
    lr_lo = max(0.001, opt["learning_rate"] * 0.5)
    lr_hi = min(0.3, opt["learning_rate"] * 1.5)
    md_lo = max(3, opt["max_depth"] - 3)
    md_hi = min(24, opt["max_depth"] + 3)
    subs_lo = max(0.4, opt["subsample"] - 0.2)
    subs_hi = min(1.0, opt["subsample"] + 0.2)
    col_lo = max(0.4, opt["colsample_bytree"] - 0.2)
    col_hi = min(1.0, opt["colsample_bytree"] + 0.2)
    nest_lo = max(100, opt["n_estimators"] - 200)
    nest_hi = min(1200, opt["n_estimators"] + 200)
    return [(lr_lo, lr_hi), (md_lo, md_hi), (subs_lo, subs_hi), (col_lo, col_hi), (nest_lo, nest_hi)]

de_bounds = de_bounds_from_opt(optuna_best)

def de_obj(vec):
    try:
        lr, md, subs, col, nest = vec
        p = dict(optuna_best)
        p.update({
            "learning_rate": float(np.clip(lr, 0.001, 0.3)),
            "max_depth": int(np.clip(round(md), 3, 24)),
            "subsample": float(np.clip(subs, 0.4, 1.0)),
            "colsample_bytree": float(np.clip(col, 0.4, 1.0)),
            "n_estimators": int(np.clip(round(nest), 50, 2000))
        })
        score = xgb_cv_score(p, X, y, folds=3)
        # Print first evaluation similar to stable
        # differential_evolution internal call prints are not available,
        # but we print the first successful score outside loop below.
        return -float(score)
    except Exception:
        return 1.0  # worst penalty

res = differential_evolution(de_obj, de_bounds, maxiter=6, popsize=6, seed=SEED, polish=False)
de_time = time.time() - de_start
de_refined = {
    **optuna_best,
    "learning_rate": float(np.clip(res.x[0], 0.001, 0.3)),
    "max_depth": int(np.clip(round(res.x[1]), 3, 24)),
    "subsample": float(np.clip(res.x[2], 0.4, 1.0)),
    "colsample_bytree": float(np.clip(res.x[3], 0.4, 1.0)),
    "n_estimators": int(np.clip(round(res.x[4]), 50, 2000))
}
print("DE refined params:", de_refined)

# ---------------- 3) GA + DE Hybrid (stable flow) ----------------
print("\n=== GA + DE Hybrid Optimization ===")
ga_start = time.time()
def evaluate_candidate_with_stats(params, folds=3):
    # returns mean, std
    params = clamp(safe_cast(params))
    skf_local = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    accs = []
    for tr,te in skf_local.split(X, y):
        Xtr = X.iloc[tr].copy(); Xte = X.iloc[te].copy()
        ytr = y[tr].copy(); yte = y[te].copy()
        missing = np.setdiff1d(np.arange(n_classes), np.unique(ytr))
        if len(missing):
            dummy = np.zeros((len(missing), Xtr.shape[1]))
            Xtr_v = np.vstack([Xtr.values, dummy])
            ytr_v = np.concatenate([ytr, missing])
        else:
            Xtr_v = Xtr.values; ytr_v = ytr
        try:
            m = make_xgb(params)
            m.fit(Xtr_v, ytr_v)
            preds = m.predict(Xte.values)
            accs.append(accuracy_score(yte, preds))
        except Exception:
            accs.append(0.0)
    arr = np.array(accs)
    return float(arr.mean()), float(arr.std())

def ga_de_hybrid(seed_params, pop_size=20, generations=10, mutation_scale=0.12, elite_k=3, folds=3):
    bounds = {
        "learning_rate": (0.005, 0.06),
        "max_depth": (4, 24),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.4, 1.0),
        "n_estimators": (200, 1200),
        "min_child_weight": (1.0, 6.0),
        "gamma": (0.0, 2.0),
        "reg_lambda": (0.5, 6.0)
    }
    seed_params = clamp(safe_cast(seed_params))
    def rand_near(base, scale=0.08):
        out = {}
        for k,(lo,hi) in bounds.items():
            v = float(base.get(k, (lo+hi)/2))
            span = hi-lo
            out[k] = float(np.clip(v + np.random.normal(0, scale*span), lo, hi))
        return out

    pop = [seed_params.copy()]
    for _ in range(pop_size-1):
        pop.append(rand_near(seed_params))

    fitness_cache = {}
    def eval_cache(p):
        key = tuple(sorted([(k, float(p[k])) for k in sorted(p.keys())]))
        if key in fitness_cache: return fitness_cache[key]
        mean, std = evaluate_candidate_with_stats(p, folds=folds)
        fitness = mean - 0.001 * std  # stable fitness formula
        fitness_cache[key] = (fitness, mean, std)
        return fitness, mean, std

    for g in range(generations):
        scored = []
        for p in pop:
            fit, mean, std = eval_cache(p)
            scored.append((fit, mean, std, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        print(f"Gen {g} best={best[1]:.6f} (fitness {best[0]:.6f}, std {best[2]:.6f})")

        elites = [copy.deepcopy(t[3]) for t in scored[:elite_k]]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1 = random.choice(elites); p2 = random.choice(elites)
            child = {}
            for k,(lo,hi) in bounds.items():
                child[k] = p1[k] if random.random() < 0.5 else p2[k]
                span = hi-lo
                child[k] = float(np.clip(child[k] + np.random.normal(0, mutation_scale*span), lo, hi))
            new_pop.append(child)

        # DE-like local moves occasionally
        if g % 3 == 0:
            for _ in range(3):
                r1, r2 = random.sample(pop, 2)
                mutant = {}
                F = 0.8
                for k,(lo,hi) in bounds.items():
                    v = best[3][k] + F * (r1[k] - r2[k]) + np.random.normal(0, 0.03*(hi-lo))
                    mutant[k] = float(np.clip(v, lo, hi))
                new_pop.append(mutant)
                if len(new_pop) >= pop_size: break

        pop = new_pop[:pop_size]

    # finalize: compute top-3 average
    final_scored = []
    for p in pop:
        fit, mean, std = eval_cache(p)
        final_scored.append((fit, mean, std, p))
    final_scored.sort(key=lambda x: x[0], reverse=True)
    top3 = final_scored[:3]
    print("GA+DE top results (mean acc):", [t[1] for t in top3])
    # aggregate top3
    agg = {}
    for k in top3[0][3].keys():
        vals = [t[3][k] for t in top3]
        if k in ("max_depth", "n_estimators"):
            agg[k] = int(round(np.mean(vals)))
        else:
            agg[k] = float(np.mean(vals))
    agg_score = float(np.mean([t[1] for t in top3]))
    return agg, agg_score, top3

hybrid_params, hybrid_score, hybrid_top3 = ga_de_hybrid(de_refined, pop_size=20, generations=10, mutation_scale=0.12, elite_k=3, folds=3)
ga_time = time.time() - ga_start
print("Hybrid refined params:", hybrid_params, "score:", hybrid_score)

# choose tuned params (prefer hybrid if score >= de score)
de_score = xgb_cv_score(de_refined, X, y, folds=3)
if hybrid_score >= de_score:
    XGB_TUNED_PARAMS = hybrid_params
else:
    XGB_TUNED_PARAMS = de_refined

print("\n✅ XGB_TUNED_PARAMS ready for baseline CV:", XGB_TUNED_PARAMS)

# --- Ablation study results ---
ablation_results = {
    "Optuna": optuna_score,
    "Optuna+DE": de_score,
    "Full_EvoX": hybrid_score
}

pd.DataFrame(list(ablation_results.items()),
             columns=["stage","accuracy"]
).to_csv("./output/ablation_results.csv", index=False)

# --- Computational efficiency results ---
eff_df = pd.DataFrame({
    "method": ["Optuna", "Optuna+DE", "EvoX"],
    "time_seconds": [optuna_time, de_time, ga_time]
})

eff_df.to_csv("./output/optimization_runtime.csv", index=False)

# === Save tuned EvoX-Boost parameters (for dashboard SHAP & reproducibility) ===
import pickle

json.dump(
    {k: float(v) if not isinstance(v, (int, np.integer)) else int(v) 
     for k, v in XGB_TUNED_PARAMS.items()},
    open("./output/xgb_tuned_params.json", "w"),
    indent=4
)

# === Train final XGB model on full dataset & save it for SHAP ===
final_xgb = make_xgb(clamp(safe_cast(XGB_TUNED_PARAMS)))
final_xgb.fit(X.values, y)
pickle.dump(final_xgb, open("./output/xgb_model.pkl","wb"))
print("Saved xgb_tuned_params.json and xgb_model.pkl")

# --- Hyperparameter sensitivity ---
print("\n=== Hyperparameter Sensitivity Analysis ===")

lr_values = [0.01,0.02,0.03,0.05]
sens_results = []

for lr in lr_values:

    p = dict(XGB_TUNED_PARAMS)
    p["learning_rate"] = lr

    acc = xgb_cv_score(p, X, y, folds=3)

    sens_results.append({
        "parameter":"learning_rate",
        "value":lr,
        "accuracy":acc
    })

pd.DataFrame(sens_results).to_csv("./output/sensitivity_results.csv", index=False)

# --- Robustness experiment ---
print("\n=== Robustness Test ===")

robust_results = []

# original
orig_acc = xgb_cv_score(XGB_TUNED_PARAMS, X, y, folds=3)
robust_results.append({"scenario":"original","accuracy":orig_acc})

# noise
noise = np.random.normal(0,0.01,X.shape)
X_noise = X + noise

noise_acc = xgb_cv_score(XGB_TUNED_PARAMS, X_noise, y, folds=3)
robust_results.append({"scenario":"noise_0.01","accuracy":noise_acc})

# missing simulation
X_missing = X.copy()
mask = np.random.rand(*X_missing.shape) < 0.05
X_missing[mask] = np.nan
X_missing = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X_missing),
                         columns=X.columns)

miss_acc = xgb_cv_score(XGB_TUNED_PARAMS, X_missing, y, folds=3)
robust_results.append({"scenario":"5pct_missing","accuracy":miss_acc})

pd.DataFrame(robust_results).to_csv("./output/robustness_results.csv", index=False)

# ---------------- final evaluators (stable style) ----------------
def eval_rf(Xtr, ytr, Xte, yte, params=None):
    try:
        if GPU_CUML:
            gXtr = cudf.DataFrame.from_pandas(Xtr); gXte = cudf.DataFrame.from_pandas(Xte)
            gytr = cudf.Series(ytr.astype(np.int32))
            m = make_rf(params); m.fit(gXtr, gytr); preds = m.predict(gXte).to_numpy()
        else:
            m = make_rf(params); m.fit(Xtr, ytr); preds = m.predict(Xte)
    except Exception:
        m = make_rf(params); m.fit(Xtr, ytr); preds = m.predict(Xte)
    return float(accuracy_score(yte, preds)), np.asarray(preds, dtype=int)

def eval_xgb_final(params, Xtr, ytr, Xte, yte):
    p = clamp(safe_cast(params))
    m = make_xgb(p)
    missing = np.setdiff1d(np.arange(n_classes), np.unique(ytr))
    if len(missing):
        dummy = np.zeros((len(missing), Xtr.shape[1]))
        Xtr_aug = np.vstack([Xtr.values, dummy]); ytr_aug = np.concatenate([ytr, missing])
    else:
        Xtr_aug = Xtr.values; ytr_aug = ytr
    m.fit(Xtr_aug, ytr_aug)
    try:
        proba = m.predict_proba(Xte.values)
    except Exception:
        proba = None
    preds = m.predict(Xte.values)
    return float(accuracy_score(yte, preds)), np.asarray(preds, dtype=int), proba

def eval_lgb(Xtr, ytr, Xte, yte):
    try:
        m = make_lgb(); m.fit(Xtr.values, ytr)
        preds = m.predict(Xte.values)
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=200, random_state=SEED); m.fit(Xtr, ytr); preds = m.predict(Xte)
    return float(accuracy_score(yte, preds)), np.asarray(preds, dtype=int)

# ---------------- 5-fold CV & SAVE (stable style) ----------------
print("\n=== FINAL 5-FOLD CV & SAVE ===")
rf_scores = []; xgb_scores = []; lgb_scores = []
if os.path.exists("./output/fold_runtime_log.txt"):
    os.remove("./output/fold_runtime_log.txt")

for f, (tr, te) in enumerate(skf.split(X, y), 1):
    print(f"\n===== FOLD {f}/{N_FOLDS} =====")
    t0 = time.time()
    Xtr = X.iloc[tr].reset_index(drop=True); Xte = X.iloc[te].reset_index(drop=True)
    ytr = y[tr]; yte = y[te]

    # RF
    try:
        rf_acc, rf_preds = eval_rf(Xtr, ytr, Xte, yte)
        rf_scores.append(rf_acc)
        pd.DataFrame({"fold":[f]*len(yte), "y_true": yte, "y_pred": rf_preds}).to_csv(f"./output/rf_pred_fold_{f}.csv", index=False)
        print("RF ✓", rf_acc)
    except Exception as e:
        print("RF fail:", e)

    # XGB
    try:
        xgb_acc, xgb_preds, xgb_proba = eval_xgb_final(XGB_TUNED_PARAMS, Xtr, ytr, Xte, yte)
        xgb_scores.append(xgb_acc)
        prob_max = (xgb_proba.max(axis=1) if xgb_proba is not None else [None]*len(yte))
        # Build full prob dataframe
        if xgb_proba is not None:
            proba_df = pd.DataFrame(
                xgb_proba,
                columns=[f"prob_{i}" for i in range(xgb_proba.shape[1])]
            )
        else:
            proba_df = pd.DataFrame()

        outdf = pd.DataFrame({
            "fold": [f]*len(yte),
            "y_true": yte,
            "y_pred": xgb_preds
        })

        # Attach prob columns if available
        outdf = pd.concat([outdf.reset_index(drop=True),
                           proba_df.reset_index(drop=True)],
                        axis=1)

        outdf.to_csv(f"./output/xgb_pred_fold_{f}.csv", index=False)

        print("XGB ✓", xgb_acc)
    except Exception as e:
        print("XGB fail:", e)

    # LGB (baseline, suppressed verbosity)
    try:
        lgb_acc, lgb_preds = eval_lgb(Xtr, ytr, Xte, yte)
        lgb_scores.append(lgb_acc)
        print("LGB ✓", lgb_acc)
    except Exception as e:
        print("LGB fail:", e)

    dt = time.time() - t0
    with open("./output/fold_runtime_log.txt", "a") as log:
        log.write(f"Fold {f}: time={dt:.3f}s\n")
    print(f"⏱ Fold {f} runtime = {dt:.2f}s")

# ---------------- SUMMARY ----------------
print("\n=== Final Summary ===")
rf_mean = float(np.nanmean(rf_scores)) if len(rf_scores) else 0.0
xgb_mean = float(np.nanmean(xgb_scores)) if len(xgb_scores) else 0.0
lgb_mean = float(np.nanmean(lgb_scores)) if len(lgb_scores) else 0.0

print(f"cuML_RF  mean: {rf_mean:.4f}")
print(f"XGBoost  mean: {xgb_mean:.4f}  (best: {np.nanmax(xgb_scores) if len(xgb_scores) else 0.0:.4f})")
print(f"LightGBM mean: {lgb_mean:.4f}")
overall_mean = np.nanmean([rf_mean, xgb_mean, lgb_mean])
print(f"Overall Mean Accuracy (model-average) ≈ {overall_mean:.4f}")

summary = pd.DataFrame({"RF": rf_scores, "XGB": xgb_scores, "LGB": lgb_scores})
summary.loc["Mean"] = summary.mean()
summary.to_csv("./output/extreme_run_gpu_summary.csv")
print(f"\n✅ Results saved to ./output/extreme_run_gpu_summary.csv")
# --- Statistical significance testing ---
from scipy.stats import friedmanchisquare, wilcoxon

print("\n=== Statistical Significance Testing ===")

# Convert fold scores to arrays
rf_arr = np.array(rf_scores)
xgb_arr = np.array(xgb_scores)
lgb_arr = np.array(lgb_scores)

# Ensure equal length for tests
min_len = min(len(rf_arr), len(xgb_arr), len(lgb_arr))
rf_arr = rf_arr[:min_len]
xgb_arr = xgb_arr[:min_len]
lgb_arr = lgb_arr[:min_len]

# -------- Friedman Test --------
stat, p = friedmanchisquare(rf_arr, xgb_arr, lgb_arr)

print(f"Friedman statistic : {stat:.4f}")
print(f"Friedman p-value   : {p:.6f}")

# -------- Pairwise Wilcoxon Tests --------
w_rf = wilcoxon(xgb_arr, rf_arr).pvalue
w_lgb = wilcoxon(xgb_arr, lgb_arr).pvalue

print(f"EvoX vs RF p-value        : {w_rf:.6f}")
print(f"EvoX vs LightGBM p-value  : {w_lgb:.6f}")

# Save statistical test results
stats_df = pd.DataFrame({
    "comparison": ["EvoX vs RF", "EvoX vs LightGBM"],
    "p_value": [w_rf, w_lgb]
})

stats_df.to_csv("./output/statistical_tests.csv", index=False)
print("DONE.")