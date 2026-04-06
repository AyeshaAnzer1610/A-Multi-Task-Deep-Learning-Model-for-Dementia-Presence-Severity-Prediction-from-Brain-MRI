"""
src/evaluate.py  —  Full evaluation pipeline (paper §3.9, §4.5)
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    balanced_accuracy_score,
)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    p_bin, p_ge1, p_ge2 = [], [], []
    y_bin, y_ord         = [], []
    subj_ids, paths      = [], []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        out  = model(imgs)
        p_bin.append(out["binary"].cpu().float().numpy())
        p_ge1.append(out["ordinal1"].cpu().float().numpy())
        p_ge2.append(out["ordinal2"].cpu().float().numpy())
        y_bin.append(batch["binary"].numpy())
        y_ord.append(batch["ordinal"].numpy())
        subj_ids.extend(batch["subject_id"].numpy().tolist())
        paths.extend(batch["path"])

    return {
        "p_binary":    np.concatenate(p_bin),
        "p_ge1":       np.concatenate(p_ge1),
        "p_ge2":       np.concatenate(p_ge2),
        "y_binary":    np.concatenate(y_bin).astype(int),
        "y_ordinal":   np.concatenate(y_ord).astype(int),
        "subject_ids": np.array(subj_ids),
        "paths":       paths,
    }


def tune_thresholds(preds, samples):
    subj = _aggregate_subject_level(preds)

    best_f1, best_t1, best_t2 = -1.0, 0.45, 0.55
    for t1 in np.arange(0.20, 0.81, 0.10):
        for t2 in np.arange(0.20, 0.81, 0.10):
            yh = _decode_ordinal_np(subj["p_ge1"], subj["p_ge2"],
                                    float(t1), float(t2))
            f1 = f1_score(subj["y_ordinal"], yh, average="macro",
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, float(t1), float(t2)

    for t1 in np.arange(max(0.05, best_t1-0.09), min(0.95, best_t1+0.10), 0.01):
        for t2 in np.arange(max(0.05, best_t2-0.09), min(0.95, best_t2+0.10), 0.01):
            yh = _decode_ordinal_np(subj["p_ge1"], subj["p_ge2"],
                                    float(t1), float(t2))
            f1 = f1_score(subj["y_ordinal"], yh, average="macro",
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, float(t1), float(t2)

    best_ba, best_tbin = -1.0, 0.50
    for tbin in np.arange(0.05, 0.96, 0.005):
        yh   = (subj["p_binary"] >= tbin).astype(int)
        bacc = balanced_accuracy_score(subj["y_binary"], yh)
        if bacc > best_ba:
            best_ba, best_tbin = bacc, float(tbin)

    return best_t1, best_t2, best_tbin


def compute_image_level_metrics(preds, t1, t2, tbin):
    yhat_bin = (preds["p_binary"] >= tbin).astype(int)
    yhat_ord = _decode_ordinal_np(preds["p_ge1"], preds["p_ge2"], t1, t2)
    return {
        "binary":   _metrics(preds["y_binary"],  yhat_bin,
                             preds["p_binary"], "binary"),
        "severity": _metrics(preds["y_ordinal"], yhat_ord,
                             np.stack([preds["p_ge1"], preds["p_ge2"]], 1),
                             "ordinal"),
    }


def _aggregate_subject_level(preds):
    buckets = defaultdict(lambda: {k: [] for k in
                ["p_binary","p_ge1","p_ge2","y_binary","y_ordinal"]})
    for i, sid in enumerate(preds["subject_ids"]):
        b = buckets[int(sid)]
        for k in ["p_binary","p_ge1","p_ge2","y_binary","y_ordinal"]:
            b[k].append(preds[k][i])

    out = {k: [] for k in
           ["subject_ids","p_binary","p_ge1","p_ge2","y_binary","y_ordinal"]}
    for sid, b in buckets.items():
        out["subject_ids"].append(sid)
        out["p_binary"].append(float(np.mean(b["p_binary"])))
        m1 = float(np.mean(b["p_ge1"]))
        m2 = min(float(np.mean(b["p_ge2"])), m1)
        out["p_ge1"].append(m1)
        out["p_ge2"].append(m2)
        out["y_binary"].append(int(round(np.mean(b["y_binary"]))))
        out["y_ordinal"].append(int(round(np.mean(b["y_ordinal"]))))

    return {k: np.array(v) for k, v in out.items()}


def compute_subject_level_metrics(preds, samples, t1, t2, tbin):
    subj     = _aggregate_subject_level(preds)
    yhat_bin = (subj["p_binary"] >= tbin).astype(int)
    yhat_ord = _decode_ordinal_np(subj["p_ge1"], subj["p_ge2"], t1, t2)
    return {
        "binary":   _metrics(subj["y_binary"],  yhat_bin,
                             subj["p_binary"], "binary"),
        "severity": _metrics(subj["y_ordinal"], yhat_ord,
                             np.stack([subj["p_ge1"], subj["p_ge2"]], 1),
                             "ordinal"),
    }


def bootstrap_ci(y_true, y_pred, y_prob, task="binary",
                 n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    accs, aucs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        try:
            if task == "binary":
                aucs.append(roc_auc_score(yt, ypr))
            else:
                pr = _ordinal_class_probs(ypr)
                aucs.append(roc_auc_score(yt, pr, multi_class="ovr",
                                          average="macro"))
        except ValueError:
            pass

    def ci(arr):
        return (float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))
    return {"accuracy": ci(accs), "auc": ci(aucs), "macro_f1": ci(f1s)}


def mcnemar_test(y_true, y_pred_proposed, y_pred_baseline):
    cp = (y_pred_proposed == y_true)
    cb = (y_pred_baseline == y_true)
    b  = int(np.sum(cp & ~cb))
    c  = int(np.sum(~cp & cb))
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mn
        res = sm_mn([[0, b],[c, 0]], exact=(b+c) < 25, correction=True)
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
                "significant": bool(res.pvalue < 0.05), "b": b, "c": c}
    except ImportError:
        from scipy.stats import chi2
        stat = (abs(b-c)-1.0)**2 / max(b+c, 1)
        pval = 1 - chi2.cdf(stat, df=1)
        return {"statistic": float(stat), "pvalue": float(pval),
                "significant": bool(pval < 0.05), "b": b, "c": c}


def full_evaluation_report(model, test_loader, test_samples, device,
                            t1, t2, tbin,
                            baseline_bin_preds=None,
                            baseline_sev_preds=None,
                            n_boot=1000):
    print("\n" + "="*65)
    print("  EVALUATION REPORT")
    print("="*65)

    preds    = collect_predictions(model, test_loader, device)
    il       = compute_image_level_metrics(preds, t1, t2, tbin)
    subj     = _aggregate_subject_level(preds)
    yhat_bin = (subj["p_binary"] >= tbin).astype(int)
    yhat_ord = _decode_ordinal_np(subj["p_ge1"], subj["p_ge2"], t1, t2)

    sl_bin = _metrics(subj["y_binary"],  yhat_bin,
                      subj["p_binary"], "binary")
    sl_sev = _metrics(subj["y_ordinal"], yhat_ord,
                      np.stack([subj["p_ge1"], subj["p_ge2"]], 1), "ordinal")

    print(f"\n  {'Task':<28} {'Level':<10} {'Acc':>6} {'AUC':>6} {'F1':>6}")
    print("  " + "-"*57)
    rows = [("Binary Detection",  il["binary"],    "Image"),
            ("Binary Detection",  sl_bin,          "Subject"),
            ("Severity Staging",  il["severity"],  "Image"),
            ("Severity Staging",  sl_sev,          "Subject")]
    for task, m, lv in rows:
        print(f"  {task:<28} {lv:<10} "
              f"{m['accuracy']:>6.4f} {m['auc']:>6.4f} {m['macro_f1']:>6.4f}")

    print("\n  Confusion Matrix — Binary (subject-level):")
    print(confusion_matrix(subj["y_binary"], yhat_bin))
    print("\n  Confusion Matrix — Severity (subject-level):")
    print(confusion_matrix(subj["y_ordinal"], yhat_ord))
    print("\n  Per-class Severity Report:")
    print(classification_report(subj["y_ordinal"], yhat_ord,
          target_names=["Non-Dem.","Very Mild","Mild/Mod."], zero_division=0))

    print(f"\n  Bootstrap 95% CI ({n_boot} resamples):")
    ci_bin = bootstrap_ci(subj["y_binary"],  yhat_bin, subj["p_binary"],
                          "binary",  n_boot)
    ci_sev = bootstrap_ci(subj["y_ordinal"], yhat_ord,
                          np.stack([subj["p_ge1"], subj["p_ge2"]], 1),
                          "ordinal", n_boot)
    for label, val, ci in [
        ("Binary  AUC",  sl_bin["auc"],       ci_bin["auc"]),
        ("Binary  Acc",  sl_bin["accuracy"],  ci_bin["accuracy"]),
        ("Binary  F1 ",  sl_bin["macro_f1"],  ci_bin["macro_f1"]),
        ("Severity AUC", sl_sev["auc"],       ci_sev["auc"]),
        ("Severity Acc", sl_sev["accuracy"],  ci_sev["accuracy"]),
        ("Severity F1 ", sl_sev["macro_f1"],  ci_sev["macro_f1"]),
    ]:
        print(f"    {label}: {val:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    if baseline_bin_preds is not None:
        mn  = mcnemar_test(subj["y_binary"],  yhat_bin, baseline_bin_preds)
        sig = "significant" if mn["significant"] else "NOT significant"
        print(f"\n  McNemar binary  : p={mn['pvalue']:.4f}  ({sig})")
    if baseline_sev_preds is not None:
        mn  = mcnemar_test(subj["y_ordinal"], yhat_ord, baseline_sev_preds)
        sig = "significant" if mn["significant"] else "NOT significant"
        print(f"  McNemar severity: p={mn['pvalue']:.4f}  ({sig})")

    errs    = subj["y_ordinal"] != yhat_ord
    adj     = errs & (np.abs(subj["y_ordinal"] - yhat_ord) == 1)
    non_adj = errs & (np.abs(subj["y_ordinal"] - yhat_ord)  > 1)
    n_s     = len(subj["y_ordinal"])
    print(f"\n  Ordinal error analysis (n={n_s} subjects):")
    print(f"    Adjacent errors    : {adj.sum()}  "
          f"({100*adj.sum()/max(errs.sum(),1):.1f}% of errors)")
    print(f"    Non-adjacent errors: {non_adj.sum()}  "
          f"({100*non_adj.sum()/max(errs.sum(),1):.1f}% of errors)")
    print(f"    Correct or adjacent: {(~errs|adj).sum()}/{n_s} = "
          f"{100*(~errs|adj).mean():.1f}%")
    print("\n" + "="*65)

    return {
        "il": il, "sl_binary": sl_bin, "sl_severity": sl_sev,
        "ci_binary": ci_bin, "ci_severity": ci_sev,
        "subject_predictions": {
            "y_binary":  subj["y_binary"],  "y_ordinal": subj["y_ordinal"],
            "yhat_bin":  yhat_bin,           "yhat_ord":  yhat_ord,
            "p_binary":  subj["p_binary"],   "p_ge1":     subj["p_ge1"],
            "p_ge2":     subj["p_ge2"],
        },
        "preds": preds,
    }


def _decode_ordinal_np(p_ge1, p_ge2, t1, t2):
    pred = np.zeros(len(p_ge1), dtype=int)
    pred[p_ge1 >= t1] = 1
    pred[(p_ge1 >= t1) & (p_ge2 >= t2)] = 2
    return pred


def _ordinal_class_probs(p_cumulative):
    if p_cumulative.ndim == 1:
        return p_cumulative
    p0  = np.clip(1.0 - p_cumulative[:, 0], 0, 1)
    p1  = np.clip(p_cumulative[:, 0] - p_cumulative[:, 1], 0, 1)
    p2  = np.clip(p_cumulative[:, 1], 0, 1)
    mat = np.stack([p0, p1, p2], axis=1)
    return mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-9)


def _metrics(y_true, y_pred, y_prob, task):
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        if task == "binary":
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            pr  = _ordinal_class_probs(y_prob)
            auc = float(roc_auc_score(y_true, pr, multi_class="ovr",
                                      average="macro"))
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc, "macro_f1": f1}
