"""End-to-end experiment pipeline.

    data -> preprocessing -> {binary, multiclass} x {4 models} -> metrics -> figures

Evaluation protocol
-------------------
1. **In-distribution validation**: each model is trained on a stratified 80 %
   of KDDTrain+ and scored on the held-out 20 %. This is the number most
   tutorials report and it is optimistic.
2. **Official test**: each model is re-trained on 100 % of KDDTrain+ and scored
   on KDDTest+ (different distribution, 17 attack types unseen in training)
   and on its hard subset KDDTest-21 (records that fewer than 21 of the
   dataset authors' 21 reference classifiers got right, i.e. difficulty < 21).
3. **Ablation**: multi-class models re-trained with ``class_weight="balanced"``.
4. **Novel-attack analysis**: binary detection rate on attack types that do
   and do not occur in the training data.
5. **Feature importance**: impurity importance and test-set permutation
   importance of the binary random forest.

All outputs are written to ``results/`` (or ``--out``).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score as auc_fn
from sklearn.model_selection import train_test_split

from . import config, plots
from .data import load_dataset
from .evaluation import binary_metrics, confusion, multiclass_metrics
from .models import MODEL_LABELS, MODEL_NAMES, make_pipeline
from .preprocessing import split_xy

log = logging.getLogger("nids")

REAL_MODELS = [m for m in MODEL_NAMES if m != "majority"]


def _scores(model, X, task):
    """Predictions plus probability outputs for one fitted pipeline."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    if task == "binary":
        return pred, proba[:, list(model.classes_).index(1)]
    return pred, proba


def _metrics(task, y, pred, score, classes):
    if task == "binary":
        return binary_metrics(y, pred, score)
    return multiclass_metrics(y, pred, score, classes)


def run(out_dir: Path = config.RESULTS_DIR, force_synthetic: bool = False,
        perm_repeats: int = 5, perm_sample: int = 5000) -> dict:
    t_start = time.time()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ds = load_dataset(force_synthetic=force_synthetic)
    train, test = ds.train, ds.test
    test21 = test[test["difficulty"] < 21]
    eval_sets = {"KDDTest+": test, "KDDTest-21": test21}
    train_attacks = set(train["attack"])
    novel_attacks = sorted(set(test["attack"]) - train_attacks)
    log.info("Source=%s train=%d test=%d test21=%d novel attack types=%d",
             ds.source, len(train), len(test), len(test21), len(novel_attacks))

    # Dataset summary tables.
    summary = pd.DataFrame({
        "train": train["category"].value_counts(),
        "test_plus": test["category"].value_counts(),
        "test_21": test21["category"].value_counts(),
    }).reindex(config.CATEGORY_ORDER).fillna(0).astype(int)
    summary.loc["total"] = summary.sum()
    summary.to_csv(out_dir / "dataset_summary.csv", index_label="category")
    per_attack = pd.DataFrame({
        "category": pd.concat([train, test]).groupby("attack")["category"].first(),
        "train": train["attack"].value_counts(),
        "test_plus": test["attack"].value_counts(),
    }).fillna(0)
    per_attack[["train", "test_plus"]] = per_attack[["train", "test_plus"]].astype(int)
    per_attack["novel"] = per_attack.index.isin(novel_attacks)
    per_attack.sort_values(["category", "train"], ascending=[True, False]).to_csv(
        out_dir / "attack_type_counts.csv", index_label="attack")
    plots.plot_class_distribution(train, test, fig_dir / "class_distribution.png")

    # Stratified 80/20 split of the training data for in-distribution validation.
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.2,
                                      stratify=train["category"], random_state=config.RANDOM_STATE)
    train80, val20 = train.iloc[tr_idx], train.iloc[va_idx]

    rows, fitted, roc_data, reports = [], {}, {k: {} for k in eval_sets}, []
    roc_mc: dict = {}
    for task in ("binary", "multiclass"):
        X_tr, y_tr = split_xy(train, task)
        for name in MODEL_NAMES:
            # (1) in-distribution validation
            m_val = make_pipeline(name)
            m_val.fit(*split_xy(train80, task))
            X_va, y_va = split_xy(val20, task)
            pred, score = _scores(m_val, X_va, task)
            rows.append({"task": task, "model": name, "class_weight": "none",
                         "eval_set": "KDDTrain+ 20% hold-out",
                         **_metrics(task, y_va, pred, score, m_val.classes_)})

            # (2) full training set -> official test sets
            model = make_pipeline(name)
            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_s = time.time() - t0
            fitted[(task, name)] = model
            for split, df in eval_sets.items():
                X_te, y_te = split_xy(df, task)
                pred, score = _scores(model, X_te, task)
                rows.append({"task": task, "model": name, "class_weight": "none", "eval_set": split,
                             "fit_seconds": round(fit_s, 2),
                             **_metrics(task, y_te, pred, score, model.classes_)})
                if name == "majority":
                    continue
                labels = config.BINARY_ORDER if task == "binary" else config.CATEGORY_ORDER
                y_named = np.where(y_te == 1, "attack", "normal") if task == "binary" else y_te
                p_named = np.where(pred == 1, "attack", "normal") if task == "binary" else pred
                cm = confusion(y_named, p_named, labels)
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                    out_dir / f"confusion_{task}_{name}_{split.replace('+', 'plus').replace('-', '')}.csv")
                if split == "KDDTest+":
                    plots.plot_confusion(cm, labels, f"{MODEL_LABELS[name]} - {task}, {split}",
                                         fig_dir / f"confusion_{task}_{name}.png")
                    reports.append(f"=== {task} | {MODEL_LABELS[name]} | {split} ===\n"
                                   + classification_report(y_named, p_named, labels=labels,
                                                           digits=4, zero_division=0))
                if task == "binary":
                    fpr, tpr, _ = roc_curve(y_te, score)
                    roc_data[split][name] = (fpr, tpr, rows[-1]["roc_auc"])
                elif name == "hist_gb" and split == "KDDTest+":
                    # one-vs-rest ROC per attack category for the boosted model
                    for k, c in enumerate(model.classes_):
                        fpr, tpr, _ = roc_curve(y_te == c, score[:, k])
                        roc_mc[c] = (fpr, tpr, auc_fn(y_te == c, score[:, k]))
            log.info("%-10s %-14s fit=%.1fs  %s", task, name, fit_s,
                     {k: round(v, 4) for k, v in rows[-2].items()
                      if k in ("accuracy", "f1", "macro_f1", "roc_auc")})

    # (3) ablation: class re-weighting for the multi-class task
    X_tr, y_tr = split_xy(train, "multiclass")
    for name in REAL_MODELS:
        model = make_pipeline(name, class_weight="balanced")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        for split, df in eval_sets.items():
            X_te, y_te = split_xy(df, "multiclass")
            pred, proba = _scores(model, X_te, "multiclass")
            rows.append({"task": "multiclass", "model": name, "class_weight": "balanced",
                         "eval_set": split, "fit_seconds": round(fit_s, 2),
                         **multiclass_metrics(y_te, pred, proba, model.classes_)})
        log.info("ablation   %-14s balanced fit=%.1fs macro_f1=%.4f", name, fit_s, rows[-2]["macro_f1"])

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_all.csv", index=False)
    (out_dir / "classification_reports.txt").write_text("\n".join(reports))

    # Compact comparison tables (the headline numbers).
    main = metrics[(metrics.class_weight == "none")]
    bin_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate"]
    mc_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
               "macro_roc_auc_ovr"] + [f"recall_{c}" for c in config.CATEGORY_ORDER]
    comp_bin = main[main.task == "binary"][["model", "eval_set"] + bin_cols]
    comp_mc = metrics[metrics.task == "multiclass"][["model", "class_weight", "eval_set"] + mc_cols]
    comp_bin.round(4).to_csv(out_dir / "metrics_comparison_binary.csv", index=False)
    comp_mc.round(4).to_csv(out_dir / "metrics_comparison_multiclass.csv", index=False)

    # ------------------------------------------- (4) novel vs known attacks
    X_te, _ = split_xy(test, "binary")
    attack_mask = test["is_attack"].to_numpy() == 1
    novel_mask = test["attack"].isin(novel_attacks).to_numpy()
    det_rows, per_type = [], {}
    for name in REAL_MODELS:
        pred = fitted[("binary", name)].predict(X_te)
        det_rows.append({
            "model": name,
            "known_attack_detection_rate": pred[attack_mask & ~novel_mask].mean(),
            "novel_attack_detection_rate": pred[attack_mask & novel_mask].mean() if novel_mask.any() else np.nan,
            "n_known": int((attack_mask & ~novel_mask).sum()),
            "n_novel": int((attack_mask & novel_mask).sum()),
        })
        per_type[name] = pd.Series(pred[attack_mask]).groupby(test["attack"].to_numpy()[attack_mask]).mean()
    det = pd.DataFrame(det_rows)
    det.round(4).to_csv(out_dir / "novel_vs_known_detection.csv", index=False)
    pt = pd.DataFrame(per_type)
    pt.insert(0, "n_test", test[attack_mask]["attack"].value_counts())
    pt.insert(0, "novel", pt.index.isin(novel_attacks))
    pt.insert(0, "category", pt.index.map(config.ATTACK_CATEGORY))
    pt.sort_values(["category", "n_test"], ascending=[True, False]).round(4).to_csv(
        out_dir / "detection_rate_per_attack_type.csv", index_label="attack")

    # ----------------------------------------------- (5) feature importance
    rf = fitted[("binary", "random_forest")]
    feat_names = rf.named_steps["prep"].get_feature_names_out()
    # map "cat__service_http" -> "service", "log__src_bytes" -> "src_bytes"
    orig = []
    for f in feat_names:
        base = f.split("__", 1)[1]
        cat = next((c for c in config.CATEGORICAL_FEATURES if f.startswith("cat__") and base.startswith(c + "_")), None)
        orig.append(cat or base)
    mdi = pd.Series(rf.named_steps["clf"].feature_importances_, index=orig).groupby(level=0).sum()
    X_perm, y_perm = split_xy(test, "binary")
    if len(test) > perm_sample:
        X_perm, _, y_perm, _ = train_test_split(X_perm, y_perm, train_size=perm_sample, stratify=y_perm,
                                                random_state=config.RANDOM_STATE)
    perm = permutation_importance(rf, X_perm, y_perm, scoring="roc_auc", n_repeats=perm_repeats,
                                  random_state=config.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"impurity": mdi,
                        "perm_mean": pd.Series(perm.importances_mean, index=X_perm.columns),
                        "perm_std": pd.Series(perm.importances_std, index=X_perm.columns)}).fillna(0)
    imp.sort_values("perm_mean", ascending=False).round(5).to_csv(out_dir / "feature_importance.csv",
                                                                  index_label="feature")

    # --------------------------------------------------------------- figures
    # Detection rate achievable at fixed false-alarm budgets (read off the ROC
    # curve). NOTE: this uses test labels to place the threshold, so it is an
    # oracle diagnostic of ranking quality, not a deployable operating point.
    op_rows = []
    for split, models in roc_data.items():
        for name, (fpr, tpr, _) in models.items():
            row = {"eval_set": split, "model": name}
            for budget in (0.01, 0.02, 0.05, 0.10):
                row[f"tpr_at_fpr_{budget:.2f}"] = float(tpr[fpr <= budget].max())
            op_rows.append(row)
    pd.DataFrame(op_rows).round(4).to_csv(out_dir / "operating_points_binary.csv", index=False)

    plots.plot_roc(roc_data, fig_dir / "roc_curve_binary.png",
                   "ROC curves - binary attack detection (trained on KDDTrain+)")
    plots.plot_roc({"KDDTest+": {c: roc_mc[c] for c in config.CATEGORY_ORDER if c in roc_mc}},
                   fig_dir / "roc_curve_multiclass_ovr.png",
                   "One-vs-rest ROC per class - gradient boosting, 5-class task",
                   colors=plots.CLASS_COLORS, labels={c: c for c in config.CATEGORY_ORDER})
    plots.plot_feature_importance(imp, fig_dir / "feature_importance.png")
    pcr = (comp_mc[(comp_mc.eval_set == "KDDTest+") & (comp_mc.class_weight == "none")
                   & (comp_mc.model != "majority")]
           .set_index("model")[[f"recall_{c}" for c in config.CATEGORY_ORDER]].T * 100)
    pcr.index = config.CATEGORY_ORDER
    plots.plot_grouped_bars(pcr, fig_dir / "per_class_recall_multiclass.png",
                            "Per-class recall, 5-class task (KDDTest+)", "Recall (%)",
                            plots.MODEL_COLORS, MODEL_LABELS)
    dd = det.set_index("model")[["known_attack_detection_rate", "novel_attack_detection_rate"]].T * 100
    dd.index = [f"Known attack types\n(n={det.n_known[0]:,})", f"Novel attack types\n(n={det.n_novel[0]:,})"]
    plots.plot_grouped_bars(dd, fig_dir / "novel_vs_known_detection.png",
                            "Binary detection rate on KDDTest+: known vs. novel attack types",
                            "Detection rate (%)", plots.MODEL_COLORS, MODEL_LABELS, fmt="{:.1f}")

    # -------------------------------------------------------------- run info
    info = {
        "data_source": ds.source,
        "n_train": len(train), "n_test_plus": len(test), "n_test_21": len(test21),
        "novel_attack_types": novel_attacks,
        "random_state": config.RANDOM_STATE,
        "runtime_seconds": round(time.time() - t_start, 1),
        "python": platform.python_version(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    log.info("Done in %.1fs -> %s", info["runtime_seconds"], out_dir)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="NSL-KDD ML intrusion-detection pipeline")
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR, help="output directory")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the NSL-KDD download and use the synthetic fallback dataset")
    ap.add_argument("--perm-repeats", type=int, default=5, help="permutation-importance repeats")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    run(out_dir=args.out, force_synthetic=args.synthetic, perm_repeats=args.perm_repeats)


if __name__ == "__main__":
    main()
