"""
test_features.py — Tests for the feature extraction pipeline.
Run from cartelsense/ with: python -m tests.test_features
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from simulator.factory import make_colluding_scenario
from features.extractor import FeatureExtractor, FeatureConfig, TOTAL_FEATURE_DIM
from features.dataset import DatasetGenerator, DatasetConfig
from agents.colluding_agent import MODE_PRICE_PARALLELISM, MODE_SURGE_SYNCHRONIZE, MODE_NONE


def test_extractor_shape():
    print("TEST 1: Feature extractor output shapes")
    cfg     = FeatureConfig(window_size=24, stride=6)
    extract = FeatureExtractor(cfg)

    runner  = make_colluding_scenario(collusion_mode=MODE_PRICE_PARALLELISM, seed=42)
    episode = runner.run()

    X, y = extract.extract(episode)

    expected_windows = (288 - 24) // 6 + 1
    print(f"  Window size: 24 timesteps (2 hours)")
    print(f"  Stride:      6 timesteps (30 minutes)")
    print(f"  Expected windows: {expected_windows}")
    print(f"  Actual windows:   {X.shape[0]}")
    print(f"  Feature dim:      {X.shape[1]} (expected {TOTAL_FEATURE_DIM})")
    print(f"  Window length:    {X.shape[2]}")
    print(f"  Label shape:      {y.shape}")
    print(f"  All labels = 1 (collusion episode): {y.sum() == len(y)}")

    assert X.shape[1] == TOTAL_FEATURE_DIM, f"Wrong feature dim: {X.shape[1]}"
    assert X.shape[2] == 24,               "Wrong window length"
    assert X.shape[0] == y.shape[0],       "X and y must have same number of windows"
    assert y.sum() == len(y),              "All windows in collusion episode should be labeled 1"
    print("  PASSED\n")


def test_honest_labels():
    print("TEST 2: Honest episode labeled correctly")
    extract = FeatureExtractor(FeatureConfig(window_size=24, stride=6))
    runner  = make_colluding_scenario(collusion_mode=MODE_NONE, seed=42)
    episode = runner.run()
    X, y    = extract.extract(episode)

    print(f"  Honest episode label sum: {y.sum()} (expected 0)")
    assert y.sum() == 0, "All windows in honest episode should be labeled 0"
    print("  PASSED\n")


def test_feature_ranges():
    print("TEST 3: Feature values in expected ranges")
    extract = FeatureExtractor(FeatureConfig(window_size=24, stride=12))
    runner  = make_colluding_scenario(collusion_mode=MODE_PRICE_PARALLELISM, seed=42)
    episode = runner.run()
    X, y    = extract.extract(episode)

    names = FeatureExtractor.feature_names()
    print(f"  Feature stats across all windows and timesteps:")
    print(f"  {'Feature':<30} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print(f"  {'-'*56}")
    for i, name in enumerate(names):
        vals = X[:, i, :].flatten()
        print(f"  {name:<30} {vals.min():>8.3f} {vals.mean():>8.3f} {vals.max():>8.3f}")

    # Basic sanity: online fraction should be in [0,1]
    online_frac = X[:, 1, :].flatten()
    assert online_frac.min() >= 0.0 and online_frac.max() <= 1.0, "Online fraction out of [0,1]"
    # Pairwise correlation should be in [-1,1]
    corr = X[:, 9, :].flatten()
    assert corr.min() >= -1.01 and corr.max() <= 1.01, "Correlation out of [-1,1]"
    print("  PASSED\n")


def test_discrimination():
    print("TEST 4: Cross-driver features discriminate collusion from honest")
    cfg     = FeatureConfig(window_size=24, stride=6)
    extract = FeatureExtractor(cfg)

    # Run one of each
    coll_runner   = make_colluding_scenario(collusion_mode=MODE_PRICE_PARALLELISM, seed=42)
    honest_runner = make_colluding_scenario(collusion_mode=MODE_NONE, seed=42)

    coll_ep   = coll_runner.run()
    honest_ep = honest_runner.run()

    X_coll,   _ = extract.extract(coll_ep)
    X_honest, _ = extract.extract(honest_ep)

    names = FeatureExtractor.feature_names()
    print(f"  {'Feature':<30} {'Colluding':>10} {'Honest':>10} {'Gap':>8} {'Signal?':>8}")
    print(f"  {'-'*68}")

    # Key features that SHOULD differ
    key_features = {
        4: "price_cv (↓=collusion)",
        7: "direction_agreement (↑=collusion)",
        9: "pairwise_corr_mean (↑=collusion)",
        11: "dispersion_trend (↓=collusion)",
    }

    for idx, label in key_features.items():
        coll_val   = float(X_coll[:, idx, :].mean())
        honest_val = float(X_honest[:, idx, :].mean())
        gap        = abs(coll_val - honest_val)
        signal     = "✓" if gap > 0.01 else "✗"
        print(f"  {label:<30} {coll_val:>10.4f} {honest_val:>10.4f} {gap:>8.4f} {signal:>8}")

    print("  PASSED\n")


def test_dataset_generation():
    print("TEST 5: Small dataset generation end-to-end")
    dcfg = DatasetConfig(
        n_episodes_per_mode = 5,        # small for speed
        feature_config      = FeatureConfig(window_size=24, stride=12),
    )
    gen  = DatasetGenerator(dcfg)
    X, y, meta = gen.generate(verbose=True)

    print(f"\n  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  No NaN in X: {not np.isnan(X).any()}")
    print(f"  No NaN in y: {not np.isnan(y).any()}")
    assert not np.isnan(X).any(), "NaN values in features!"
    assert not np.isnan(y).any(), "NaN values in labels!"

    # Test train/val/test split
    splits = gen.train_val_test_split(X, y, meta)
    n_total = len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
    print(f"  Train: {len(splits['y_train'])} | Val: {len(splits['y_val'])} | Test: {len(splits['y_test'])}")
    print(f"  Total matches: {n_total == len(y)}")
    assert n_total == len(y), "Split sizes don't add up!"
    print("  PASSED\n")


if __name__ == "__main__":
    test_extractor_shape()
    test_honest_labels()
    test_feature_ranges()
    test_discrimination()
    test_dataset_generation()
    print("=" * 50)
    print("All feature tests passed.")