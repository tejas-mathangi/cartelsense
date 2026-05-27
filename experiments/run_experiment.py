"""
run_experiment.py — Full CartelSense experiment pipeline.

Run from cartelsense/ with:
  python -m experiments.run_experiment
"""

import sys, os, time, json
sys.path.insert(0, ".")

import numpy as np
import torch

from features.dataset import DatasetGenerator, DatasetConfig
from features.extractor import FeatureConfig
from detector.tcn import CollusionTCN
from detector.trainer import Trainer, TrainConfig
from agents.colluding_agent import (
    MODE_PRICE_PARALLELISM, MODE_ZONE_AVOIDANCE,
    MODE_SURGE_SYNCHRONIZE, MODE_PHANTOM_SCARCITY, MODE_NONE
)


def print_banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def print_metrics(label, metrics):
    print(f"  {label:<28} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | "
          f"F1={metrics['f1']:.3f} | AUC={metrics['auc']:.3f} | FPR={metrics['fpr']:.3f}")


def run():
    # ---- 1. Dataset ----
    print_banner("STEP 1: Dataset Generation")
    dcfg = DatasetConfig(
        n_episodes_per_mode=25,
        feature_config=FeatureConfig(window_size=24, stride=6),
        base_seed=0,
    )
    gen = DatasetGenerator(dcfg)
    t0  = time.time()
    X, y, meta = gen.generate(verbose=True)
    print(f"\nGeneration time: {time.time()-t0:.1f}s")

    # Balance classes
    n_honest = int((y == 0).sum())
    n_coll   = int((y == 1).sum())
    print(f"\nBefore balancing: {n_coll} collusion, {n_honest} honest")
    if n_coll > n_honest:
        rng      = np.random.default_rng(42)
        coll_idx = np.where(y == 1)[0]
        hon_idx  = np.where(y == 0)[0]
        keep     = np.sort(np.concatenate([rng.choice(coll_idx, n_honest, replace=False), hon_idx]))
        X, y, meta = X[keep], y[keep], [meta[i] for i in keep]
    print(f"After balancing:  {int((y==1).sum())} collusion, {int((y==0).sum())} honest")

    splits   = gen.train_val_test_split(X, y, meta)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    meta_test        = splits["meta_test"]
    print(f"Split: train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")

    # ---- 2. Train ----
    print_banner("STEP 2: Training TCN (all features)")
    tcfg    = TrainConfig(epochs=60, patience=10, hidden_dim=32, dropout=0.2, lr=1e-3)
    trainer = Trainer(tcfg)
    model, history = trainer.train(X_train, y_train, X_val, y_val, verbose=True)

    # ---- 3. Evaluate ----
    print_banner("STEP 3: Test Set Evaluation")
    results = trainer.evaluate(model, X_test, y_test, meta_test)

    print(f"\n  {'Metric':<28} | {'P':>5} | {'R':>5} | {'F1':>5} | {'AUC':>5} | {'FPR':>5}")
    print(f"  {'-'*65}")
    print_metrics("OVERALL", results["overall"])
    print(f"\n  Per-mode breakdown:")
    print(f"  {'-'*65}")
    labels = {
        MODE_PRICE_PARALLELISM: "Price Parallelism",
        MODE_ZONE_AVOIDANCE:    "Zone Avoidance",
        MODE_SURGE_SYNCHRONIZE: "Surge Synchronization",
        MODE_PHANTOM_SCARCITY:  "Phantom Scarcity",
        MODE_NONE:              "Honest (no collusion)",
    }
    for mode, m in sorted(results["per_mode"].items()):
        print_metrics(labels.get(mode, mode), m)

    # ---- 4. Ablation ----
    print_banner("STEP 4: Ablation Study — Feature Groups")
    ablation = {}
    for fmode in ['all', 'per_driver', 'cross_driver']:
        print(f"  Training [{fmode}]...")
        m, _ = trainer.train(X_train, y_train, X_val, y_val, verbose=False, feature_mode=fmode)
        r    = trainer.evaluate(m, X_test, y_test, meta_test, feature_mode=fmode)
        ablation[fmode] = r["overall"]

    lbl = {"all": "All features", "per_driver": "Per-driver only", "cross_driver": "Cross-driver only"}
    print(f"\n  {'Feature set':<22} | {'P':>5} | {'R':>5} | {'F1':>5} | {'AUC':>5}")
    print(f"  {'-'*45}")
    for fm, m in ablation.items():
        print(f"  {lbl[fm]:<22} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['auc']:.3f}")

    # ---- 5. Save ----
    os.makedirs("experiments/results", exist_ok=True)
    summary = {
        "overall":  results["overall"],
        "per_mode": results["per_mode"],
        "ablation": ablation,
        "history":  {k: [float(v) for v in vals] for k, vals in history.items()},
        "dataset":  {"n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test))},
    }
    with open("experiments/results/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), "experiments/results/model.pt")

    print_banner("DONE")
    print(f"  F1={results['overall']['f1']:.3f}  AUC={results['overall']['auc']:.3f}")
    print(f"  Results → experiments/results/")
    return results, ablation, history


if __name__ == "__main__":
    run()    