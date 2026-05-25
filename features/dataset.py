"""
dataset.py — Generate the labeled training dataset for the TCN detector.

Runs N episodes per collusion mode (+ N honest episodes) and extracts
windowed feature tensors from each. The result is a balanced dataset
that the detector trains on.

Dataset structure:
  X : (total_windows, TOTAL_FEATURE_DIM, window_size)  float32
  y : (total_windows,)                                  float32  {0.0, 1.0}
  meta : list of dicts with episode_id, mode, window_start per sample
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from simulator.factory import make_colluding_scenario
from features.extractor import FeatureExtractor, FeatureConfig
from agents.colluding_agent import (
    MODE_PRICE_PARALLELISM, MODE_ZONE_AVOIDANCE,
    MODE_SURGE_SYNCHRONIZE, MODE_PHANTOM_SCARCITY, MODE_NONE,
)

ALL_COLLUSION_MODES = [
    MODE_PRICE_PARALLELISM,
    MODE_ZONE_AVOIDANCE,
    MODE_SURGE_SYNCHRONIZE,
    MODE_PHANTOM_SCARCITY,
]


@dataclass
class DatasetConfig:
    """
    Controls how the training dataset is generated.

    Attributes
    ----------
    n_episodes_per_mode : episodes per collusion mode (and same for honest)
    n_honest            : honest drivers per episode
    n_colluding         : colluding drivers per episode
    grid_size           : city grid size
    base_demand         : base Poisson rate
    feature_config      : window + feature settings
    base_seed           : starting seed (each episode gets base_seed + episode_id)
    """
    n_episodes_per_mode: int           = 50
    n_honest:            int           = 6
    n_colluding:         int           = 4
    grid_size:           int           = 8
    base_demand:         float         = 3.0
    feature_config:      FeatureConfig = None
    base_seed:           int           = 0

    def __post_init__(self):
        if self.feature_config is None:
            self.feature_config = FeatureConfig()


class DatasetGenerator:
    """
    Generates the full labeled dataset by running the simulator many times.

    For each collusion mode:
      - Run n_episodes_per_mode episodes with that mode active
      - Extract windowed features from each episode
      - Label all windows as 1 (collusion)

    For honest episodes:
      - Run n_episodes_per_mode episodes with MODE_NONE
      - Label all windows as 0 (no collusion)

    Then split into train / val / test sets.
    """

    def __init__(self, config: DatasetConfig = None):
        self.config    = config or DatasetConfig()
        self.extractor = FeatureExtractor(self.config.feature_config)

    def generate(
        self,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Run all episodes and return the complete dataset.

        Returns
        -------
        X    : (N, features, window_size) float32
        y    : (N,) float32
        meta : list of dicts — {episode_id, mode, window_start, seed} per sample
        """
        cfg = self.config
        all_X    = []
        all_y    = []
        all_meta = []

        # Compute global price/demand range across a sample of episodes first
        # so normalization is consistent across the whole dataset
        if verbose:
            print("Computing normalization ranges from sample episodes...")
        price_range, demand_range = self._estimate_ranges()
        if verbose:
            print(f"  Price range:  ₹{price_range[0]:.0f} – ₹{price_range[1]:.0f}")
            print(f"  Demand range: {demand_range[0]:.2f} – {demand_range[1]:.2f}\n")

        episode_counter = 0

        # --- Collusion episodes ---
        for mode in ALL_COLLUSION_MODES:
            if verbose:
                print(f"Generating {cfg.n_episodes_per_mode} episodes: {mode}")

            for ep_idx in range(cfg.n_episodes_per_mode):
                seed   = cfg.base_seed + episode_counter
                runner = make_colluding_scenario(
                    collusion_mode = mode,
                    n_honest       = cfg.n_honest,
                    n_colluding    = cfg.n_colluding,
                    grid_size      = cfg.grid_size,
                    base_demand    = cfg.base_demand,
                    seed           = seed,
                )
                episode = runner.run(episode_id=episode_counter)

                X_ep, y_ep = self.extractor.extract(
                    episode,
                    price_range  = price_range,
                    demand_range = demand_range,
                )

                for w_idx in range(len(y_ep)):
                    all_meta.append({
                        "episode_id":   episode_counter,
                        "mode":         mode,
                        "window_start": w_idx * cfg.feature_config.stride,
                        "seed":         seed,
                    })

                all_X.append(X_ep)
                all_y.append(y_ep)
                episode_counter += 1

            if verbose:
                n_windows = sum(len(y) for y in all_y[-cfg.n_episodes_per_mode:])
                print(f"  → {n_windows} windows generated so far for this mode")

        # --- Honest episodes ---
        if verbose:
            print(f"\nGenerating {cfg.n_episodes_per_mode} honest (no-collusion) episodes")

        for ep_idx in range(cfg.n_episodes_per_mode):
            seed   = cfg.base_seed + episode_counter
            runner = make_colluding_scenario(
                collusion_mode = MODE_NONE,
                n_honest       = cfg.n_honest,
                n_colluding    = cfg.n_colluding,
                grid_size      = cfg.grid_size,
                base_demand    = cfg.base_demand,
                seed           = seed,
            )
            episode = runner.run(episode_id=episode_counter)

            X_ep, y_ep = self.extractor.extract(
                episode,
                price_range  = price_range,
                demand_range = demand_range,
            )

            for w_idx in range(len(y_ep)):
                all_meta.append({
                    "episode_id":   episode_counter,
                    "mode":         MODE_NONE,
                    "window_start": w_idx * cfg.feature_config.stride,
                    "seed":         seed,
                })

            all_X.append(X_ep)
            all_y.append(y_ep)
            episode_counter += 1

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        if verbose:
            n_pos = int(y.sum())
            n_neg = int((1 - y).sum())
            print(f"\nDataset generated:")
            print(f"  Total windows:    {len(y)}")
            print(f"  Collusion (y=1):  {n_pos} ({100*n_pos/len(y):.1f}%)")
            print(f"  Honest    (y=0):  {n_neg} ({100*n_neg/len(y):.1f}%)")
            print(f"  Feature shape:    {X.shape}")

        return X, y, all_meta

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: List[Dict],
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        seed:       int   = 42,
    ) -> Dict:
        """
        Split dataset into train / val / test.

        Important: we split by EPISODE, not by window, to prevent data leakage.
        Windows from the same episode are correlated — if they end up in both
        train and test, the model could memorize episode-specific patterns
        rather than learning generalizable collusion signals.
        """
        # Get unique episode IDs
        episode_ids = sorted(set(m["episode_id"] for m in meta))
        rng         = np.random.default_rng(seed)
        rng.shuffle(episode_ids)

        n       = len(episode_ids)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_eps = set(episode_ids[:n_train])
        val_eps   = set(episode_ids[n_train:n_train + n_val])
        test_eps  = set(episode_ids[n_train + n_val:])

        # Build index masks
        train_idx = [i for i, m in enumerate(meta) if m["episode_id"] in train_eps]
        val_idx   = [i for i, m in enumerate(meta) if m["episode_id"] in val_eps]
        test_idx  = [i for i, m in enumerate(meta) if m["episode_id"] in test_eps]

        return {
            "X_train": X[train_idx], "y_train": y[train_idx],
            "X_val":   X[val_idx],   "y_val":   y[val_idx],
            "X_test":  X[test_idx],  "y_test":  y[test_idx],
            "meta_train": [meta[i] for i in train_idx],
            "meta_val":   [meta[i] for i in val_idx],
            "meta_test":  [meta[i] for i in test_idx],
        }

    def _estimate_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Run a small number of episodes to estimate global min/max for normalization.
        Uses 5 episodes per mode to get a stable estimate.
        """
        cfg = self.config
        all_prices  = []
        all_demands = []

        for mode in ALL_COLLUSION_MODES + [MODE_NONE]:
            for i in range(5):
                runner  = make_colluding_scenario(collusion_mode=mode, seed=9999 + i)
                episode = runner.run()
                for rec in episode.timestep_records:
                    all_prices.extend(rec.driver_prices.values())
                    all_demands.extend(rec.zone_demands.values())

        return (
            (min(all_prices),  max(all_prices)),
            (min(all_demands), max(all_demands)),
        )