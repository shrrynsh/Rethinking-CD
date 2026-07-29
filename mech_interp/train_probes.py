#!/usr/bin/env python3
"""Per-layer linear probes + VCD residual-change curve for the POPE-COCO mech experiment.

Consumes the shards written by extract_activations.py and produces:

  Probe A (truth)        : hidden_state_l -> ground-truth object presence (all samples)
  Probe B (hallucination): hidden_state_l -> baseline false-positive 'Yes', restricted
                           to ground-truth-No samples only (predicts the ERROR)
  CD curve               : mean over samples of ||h_clean[l] - h_noisy[l]||_2 per layer

Outputs (under --results-dir):
  per_layer.csv          : layer, probe_truth_acc, probe_truth_auc,
                           probe_halluc_acc, probe_halluc_bal_acc, probe_halluc_auc,
                           cd_change_mean, cd_change_norm
  layerwise_curves.png   : the three normalized curves on one axis
  summary.txt / summary.json : l*_truth, l*_halluc, l*_CD, class balances, verdict

Features are standardized per fold (StandardScaler) so LogisticRegression converges;
this is a scale-only transform and does not leak labels across CV folds.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_shards(act_dir: str, splits):
    """Concatenate all shard npz across the requested splits.

    Returns clean_hidden [N,33,4096] fp32, cd_change [N,33], gt [N], pred [N], split_of [N].
    """
    ch, cd, gt, pred, split_of = [], [], [], [], []
    for split in splits:
        files = sorted(glob.glob(os.path.join(act_dir, f"{split}_shard*.npz")))
        if not files:
            print(f"[warn] no shards for split '{split}' in {act_dir}")
        for fp in files:
            z = np.load(fp)
            ch.append(z["clean_hidden"].astype(np.float32))
            cd.append(z["cd_change"].astype(np.float32))
            gt.append(z["gt"].astype(np.int64))
            pred.append(z["pred"].astype(np.int64))
            split_of.append(np.array([split] * len(z["gt"])))
    if not ch:
        raise SystemExit(f"No shards found in {act_dir} for splits {splits}")
    return (
        np.concatenate(ch),
        np.concatenate(cd),
        np.concatenate(gt),
        np.concatenate(pred),
        np.concatenate(split_of),
    )


def probe_layer(X, y, cv, seed):
    """5-fold CV accuracy + AUC + balanced-accuracy for one layer's features."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    # out-of-fold predictions for AUC / balanced acc (robust to class imbalance)
    try:
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
    except (ValueError, IndexError):
        auc = float("nan")
    yhat = cross_val_predict(clf, X, y, cv=cv)
    bal = balanced_accuracy_score(y, yhat)
    return acc, auc, bal


def run(args):
    os.makedirs(args.results_dir, exist_ok=True)
    ch, cd, gt, pred, split_of = load_shards(args.act_dir, args.splits)
    n, n_layers, dim = ch.shape
    print(f"[data] N={n}  layers={n_layers}  dim={dim}")

    # ---- class balances / sanity ----
    print("[balance] ground truth: present(Yes)={} absent(No)={}".format(int(gt.sum()), int((gt == 0).sum())))
    for split in args.splits:
        m = split_of == split
        if m.sum():
            acc = (pred[m] == gt[m]).mean()
            print(f"[baseline] {split:>12}: n={int(m.sum()):>5}  greedy acc={acc:.4f}")

    # Probe B subset: ground-truth-No only; label = model hallucinated 'Yes'
    no_mask = gt == 0
    y_hall = pred[no_mask]  # 1 = false-positive Yes
    n_no = int(no_mask.sum())
    n_hall = int(y_hall.sum())
    hall_rate = n_hall / max(n_no, 1)
    print(f"[halluc] ground-truth-No samples={n_no}  hallucinated(Yes)={n_hall}  rate={hall_rate:.4f}")
    do_probe_b = n_no >= args.min_probe_b and 0 < n_hall < n_no
    if not do_probe_b:
        print("[halluc] Probe B skipped (too few No samples or degenerate class balance).")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    rows = []
    for l in range(n_layers):
        Xl = ch[:, l, :]
        a_acc, a_auc, a_bal = probe_layer(Xl, gt, cv, args.seed)
        if do_probe_b:
            b_acc, b_auc, b_bal = probe_layer(Xl[no_mask], y_hall, cv, args.seed)
        else:
            b_acc = b_auc = b_bal = float("nan")
        rows.append(
            {
                "layer": l,
                "probe_truth_acc": a_acc,
                "probe_truth_auc": a_auc,
                "probe_truth_bal_acc": a_bal,
                "probe_halluc_acc": b_acc,
                "probe_halluc_auc": b_auc,
                "probe_halluc_bal_acc": b_bal,
                "cd_change_mean": float(cd[:, l].mean()),
            }
        )
        print(
            f"  L{l:>2}  truthAcc={a_acc:.3f} truthAUC={a_auc:.3f}  "
            f"hallAcc={b_acc:.3f} hallAUC={b_auc:.3f}  cd={cd[:, l].mean():.3f}"
        )

    df = pd.DataFrame(rows)
    cd_mean = df["cd_change_mean"].to_numpy()
    df["cd_change_norm"] = (cd_mean - cd_mean.min()) / (np.ptp(cd_mean) + 1e-12)
    csv_path = os.path.join(args.results_dir, "per_layer.csv")
    df.to_csv(csv_path, index=False)

    # ---- peaks ----
    l_truth = int(df["probe_truth_auc"].idxmax())  # AUC is imbalance-robust
    l_cd = int(df["cd_change_mean"].idxmax())
    l_hall = int(df["probe_halluc_auc"].idxmax()) if do_probe_b else None

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["probe_truth_auc"], "-o", ms=3, label="Probe A: ground-truth AUC")
    if do_probe_b:
        ax.plot(df["layer"], df["probe_halluc_auc"], "-s", ms=3, label="Probe B: hallucination AUC")
    ax.plot(df["layer"], df["cd_change_norm"], "-^", ms=3, label="VCD residual change (norm.)")
    for lyr, c, name in [(l_truth, "C0", "l*_truth"), (l_cd, "C2", "l*_CD")] + (
        [(l_hall, "C1", "l*_halluc")] if do_probe_b else []
    ):
        ax.axvline(lyr, color=c, ls="--", alpha=0.5)
    ax.set_xlabel("Layer index (0 = embeddings, 1..32 = transformer blocks)")
    ax.set_ylabel("AUC  /  normalized residual change")
    ax.set_title("POPE-COCO: where truth is decodable vs where VCD perturbs the residual stream")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.results_dir, "layerwise_curves.png")
    fig.savefig(png_path, dpi=150)

    # ---- summary ----
    summary = {
        "n_samples": int(n),
        "n_layers": int(n_layers),
        "gt_present": int(gt.sum()),
        "gt_absent": int((gt == 0).sum()),
        "probe_b_n_no_samples": n_no,
        "probe_b_hallucinated": n_hall,
        "probe_b_hallucination_rate": hall_rate,
        "probe_b_evaluated": do_probe_b,
        "l_star_truth": l_truth,
        "l_star_truth_auc": float(df.loc[l_truth, "probe_truth_auc"]),
        "l_star_halluc": l_hall,
        "l_star_halluc_auc": (float(df.loc[l_hall, "probe_halluc_auc"]) if do_probe_b else None),
        "l_star_CD": l_cd,
        "cd_at_peak": float(df.loc[l_cd, "cd_change_mean"]),
        "cd_at_truth_peak": float(df.loc[l_truth, "cd_change_mean"]),
    }
    gap_truth = abs(l_cd - l_truth)
    verdict = (
        f"l*_truth={l_truth}, l*_CD={l_cd} (gap {gap_truth} layers). "
        + ("COINCIDE" if gap_truth <= args.coincide_tol else "DIFFERENT DEPTH")
        + " -> "
        + (
            "VCD's perturbation peaks near the layer that encodes object presence."
            if gap_truth <= args.coincide_tol
            else "VCD's largest residual effect is NOT at the layer encoding object presence "
            "-- direct evidence its correction does not operate on the presence representation."
        )
    )
    if do_probe_b:
        gap_h = abs(l_cd - l_hall)
        verdict += (
            f"\nl*_halluc={l_hall}, l*_CD={l_cd} (gap {gap_h} layers). "
            + ("COINCIDE" if gap_h <= args.coincide_tol else "DIFFERENT DEPTH")
            + "."
        )
    summary["verdict"] = verdict

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write(verdict + "\n\n")
        f.write(json.dumps(summary, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    print(verdict)
    print(f"\ncsv : {csv_path}\nplot: {png_path}\nsummary: {os.path.join(args.results_dir, 'summary.txt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", default="/teamspace/studios/this_studio/cd_pope_mech_interp/activations")
    ap.add_argument("--results-dir", default="/teamspace/studios/this_studio/cd_pope_mech_interp/results")
    ap.add_argument("--splits", nargs="+", default=["random", "popular", "adversarial"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-probe-b", type=int, default=50, help="min GT-No samples to run Probe B")
    ap.add_argument("--coincide-tol", type=int, default=2, help="|peak gap| <= tol counts as coinciding")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
