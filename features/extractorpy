"""
extractor.py — Feature extraction for CartelSense.

Converts raw EpisodeData into labeled tensors ready for the TCN detector.

Two classes of features:
  A) Per-driver features   — what each individual driver is doing
  B) Cross-driver features — coordination signals across the group (the novel part)

The key insight from signal analysis:
  - Price dispersion (std dev across group) is the strongest single signal
  - Price direction agreement is huge for offline-based modes
  - Price correlation is consistent across all modes
  - These must be computed over SLIDING WINDOWS, not the full episode

Why sliding windows?
  We want the detector to make real-time decisions: "is collusion happening
  RIGHT NOW in this 1-hour window?" not just "did collusion happen today?"
  A window of W timesteps slides across the episode with stride S,
  producing many labeled samples per episode.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from simulator.episode import EpisodeData, TimestepRecord


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Controls what features are extracted and how windows are built.

    Attributes
    ----------
    window_size    : W — timesteps per window (12=1hr, 24=2hr, 48=4hr)
    stride         : how many timesteps to slide between windows
                     stride=1 → maximum overlap (best for small datasets)
                     stride=window_size → no overlap (fastest)
    use_per_driver : include individual driver features
    use_cross_driver: include cross-group coordination features
    eps            : small constant to avoid division by zero
    """
    window_size:      int   = 24      # 2-hour windows
    stride:           int   = 6       # slide every 30 minutes
    use_per_driver:   bool  = True
    use_cross_driver: bool  = True
    eps:              float = 1e-8


# ---------------------------------------------------------------------------
# What features look like (for documentation clarity)
# ---------------------------------------------------------------------------

# Per-driver features computed per timestep (before windowing):
#   [0] price_normalized     — price scaled to [0,1] using global min/max
#   [1] online               — 1.0 if online, 0.0 if offline
#   [2] local_demand_norm    — λ for driver's zone, normalized
#   [3] earnings_rate        — earnings / (price + eps), proxy for ride acceptance rate

# Cross-driver features computed per timestep across the driver group:
#   [4] price_dispersion     — std dev of group prices at this timestep (low = suspicious)
#   [5] price_mean_norm      — mean group price normalized
#   [6] online_fraction      — fraction of group online (1.0 = all online)
#   [7] direction_agreement  — fraction of drivers who moved price same direction as majority
#   [8] demand_weighted_disp — price dispersion weighted by local demand (controls for demand-driven sync)

# Window-level features (computed over the W timesteps in a window):
#   [9]  rolling_corr_mean   — mean pairwise price correlation in window
#   [10] rolling_corr_std    — std of pairwise correlations (low = all pairs correlated equally)
#   [11] dispersion_trend    — slope of price dispersion over window (decreasing = converging)
#   [12] offline_sync_score  — how synchronized are offline events in window


PER_DRIVER_DIM   = 4
CROSS_DRIVER_DIM = 5
WINDOW_LEVEL_DIM = 4
TOTAL_FEATURE_DIM = PER_DRIVER_DIM + CROSS_DRIVER_DIM + WINDOW_LEVEL_DIM  # = 13


# ---------------------------------------------------------------------------
# Core feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts features from EpisodeData and returns windowed tensors.

    Usage:
        extractor = FeatureExtractor(config=FeatureConfig(window_size=24))
        X, y = extractor.extract(episode)
        # X shape: (n_windows, n_features, window_size)
        # y shape: (n_windows,)  — binary label per window
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        episode:       EpisodeData,
        driver_ids:    Optional[List[int]] = None,
        price_range:   Optional[Tuple[float, float]] = None,
        demand_range:  Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract windowed features from one episode.

        Parameters
        ----------
        episode      : EpisodeData from simulator
        driver_ids   : which drivers to treat as "the group" being monitored
                       defaults to ALL drivers in episode
        price_range  : (min, max) for price normalization across dataset
                       if None, computed from this episode (less ideal)
        demand_range : (min, max) for demand normalization

        Returns
        -------
        X : np.ndarray, shape (n_windows, TOTAL_FEATURE_DIM, window_size)
            Feature tensor ready for TCN input
        y : np.ndarray, shape (n_windows,)
            Binary label — 1 if majority of window timesteps are collusion
        """
        cfg = self.config
        records = episode.timestep_records
        T = len(records)

        # Default: use all drivers
        if driver_ids is None:
            driver_ids = sorted(records[0].driver_prices.keys())

        # Compute normalization ranges from episode if not provided
        if price_range is None:
            all_prices = [p for rec in records for p in rec.driver_prices.values()]
            price_range = (min(all_prices), max(all_prices))
        if demand_range is None:
            all_demands = [d for rec in records for d in rec.zone_demands.values()]
            demand_range = (min(all_demands), max(all_demands))

        # Step 1: Compute per-timestep feature vectors
        # Shape: (T, TOTAL_FEATURE_DIM)
        timestep_features = self._compute_timestep_features(
            records, driver_ids, price_range, demand_range
        )

        # Step 2: Slide windows across the episode
        windows_X = []
        windows_y = []

        for start in range(0, T - cfg.window_size + 1, cfg.stride):
            end = start + cfg.window_size

            # Feature slice: (TOTAL_FEATURE_DIM, window_size) — channels-first for TCN
            window_features = timestep_features[start:end].T   # transpose to (features, time)

            # Enrich with window-level features (overwrite last WINDOW_LEVEL_DIM rows)
            window_features = self._compute_window_level_features(
                window_features, records[start:end], driver_ids, price_range
            )

            # Label: majority vote across window timesteps
            window_labels = episode.labels[start:end]
            label = int(np.mean(window_labels) >= 0.5)

            windows_X.append(window_features)
            windows_y.append(label)

        X = np.array(windows_X, dtype=np.float32)   # (n_windows, features, window_size)
        y = np.array(windows_y, dtype=np.float32)   # (n_windows,)

        return X, y

    # ------------------------------------------------------------------
    # Per-timestep features
    # ------------------------------------------------------------------

    def _compute_timestep_features(
        self,
        records:      List[TimestepRecord],
        driver_ids:   List[int],
        price_range:  Tuple[float, float],
        demand_range: Tuple[float, float],
    ) -> np.ndarray:
        """
        Compute a feature vector for each timestep.
        Returns shape: (T, TOTAL_FEATURE_DIM)
        """
        cfg = self.config
        T   = len(records)
        eps = cfg.eps

        p_min, p_max = price_range
        d_min, d_max = demand_range
        p_range      = p_max - p_min + eps
        d_range      = d_max - d_min + eps

        features = np.zeros((T, TOTAL_FEATURE_DIM), dtype=np.float32)

        prev_prices = None   # for computing price direction

        for t, rec in enumerate(records):
            prices   = np.array([rec.driver_prices.get(did, 0.0) for did in driver_ids])
            online   = np.array([float(rec.driver_online.get(did, True)) for did in driver_ids])
            earnings = np.array([rec.driver_earnings.get(did, 0.0) for did in driver_ids])

            # Local demand for each driver's zone
            demands = np.array([
                rec.zone_demands.get(rec.driver_zones.get(did, (0,0)), 0.0)
                for did in driver_ids
            ])

            # ---- Per-driver features (aggregated to group level) ----

            # [0] Mean normalized price across group
            price_norm = (prices - p_min) / p_range
            features[t, 0] = float(np.mean(price_norm))

            # [1] Fraction of group online
            features[t, 1] = float(np.mean(online))

            # [2] Mean normalized demand across group zones
            demand_norm = (demands - d_min) / d_range
            features[t, 2] = float(np.mean(demand_norm))

            # [3] Mean earnings rate (earnings / price) — proxy for acceptance rate
            acceptance = earnings / (prices + eps)
            features[t, 3] = float(np.mean(acceptance))

            # ---- Cross-driver features ----

            # [4] Price dispersion — std dev of prices across group
            #     LOW value = suspicious (everyone charging same thing)
            #     Normalized by mean price so it's comparable across demand levels
            price_std = float(np.std(prices))
            price_mean = float(np.mean(prices)) + eps
            features[t, 4] = price_std / price_mean   # coefficient of variation

            # [5] Normalized group mean price (separate from per-driver [0])
            features[t, 5] = price_mean / (p_max + eps)

            # [6] Online fraction (same as [1] but in cross-driver block for clarity)
            features[t, 6] = float(np.mean(online))

            # [7] Price direction agreement
            #     At each timestep: what fraction of drivers moved price in the majority direction?
            #     HIGH value = suspicious (everyone reacting identically)
            if prev_prices is not None:
                deltas     = prices - prev_prices
                directions = np.sign(deltas)
                majority   = np.sign(np.sum(directions))
                if majority == 0:
                    agreement = 0.5
                else:
                    agreement = float(np.mean(directions == majority))
            else:
                agreement = 0.5   # no previous timestep to compare
            features[t, 7] = agreement

            # [8] Demand-weighted price dispersion
            #     Controls for the fact that high demand naturally compresses prices
            #     (everyone raises prices during rush hour — that's honest behavior)
            #     We weight dispersion by inverse demand: high demand → expect natural sync,
            #     so we only penalize low dispersion when demand is also low.
            mean_demand = float(np.mean(demands)) + eps
            demand_weight = 1.0 / (1.0 + mean_demand)   # low demand → weight=1, high demand → weight→0
            features[t, 8] = (price_std / price_mean) * demand_weight

            # Window-level features [9:13] are filled in later per window
            # Initialize to 0 here
            features[t, 9:13] = 0.0

            prev_prices = prices.copy()

        return features

    # ------------------------------------------------------------------
    # Window-level features
    # ------------------------------------------------------------------

    def _compute_window_level_features(
        self,
        window_features: np.ndarray,              # (TOTAL_FEATURE_DIM, window_size)
        records:         List[TimestepRecord],
        driver_ids:      List[int],
        price_range:     Tuple[float, float],
    ) -> np.ndarray:
        """
        Compute features that require seeing the full window at once.
        Fills in rows [9:13] of window_features.

        These capture temporal patterns that per-timestep features miss:
          - Rolling correlation (requires multiple timesteps)
          - Dispersion trend (is the group converging in price over the window?)
          - Offline synchrony score (are offline events clustered together?)
        """
        eps = self.config.eps
        W   = len(records)

        # Extract price matrix for window: shape (n_drivers, W)
        price_matrix = np.array([
            [rec.driver_prices.get(did, 0.0) for rec in records]
            for did in driver_ids
        ], dtype=np.float32)

        # Extract online matrix: shape (n_drivers, W)
        online_matrix = np.array([
            [float(rec.driver_online.get(did, True)) for rec in records]
            for did in driver_ids
        ], dtype=np.float32)

        n_drivers = len(driver_ids)

        # Build demand matrix for residual computation
        demand_matrix = np.array([
            [records[t].zone_demands.get(records[t].driver_zones.get(did, (0,0)), 0.0)
             for t in range(W)]
            for did in driver_ids
        ], dtype=np.float32)

        # [9] Mean pairwise RESIDUAL price correlation across the window.
        #
        #     WHY RESIDUAL, NOT RAW?
        #     Raw price correlation is dominated by the shared demand signal:
        #     all drivers raise prices during rush hour → high correlation even
        #     when fully independent. This is a confounder.
        #
        #     We remove the component of each driver's price that is explained
        #     by their local demand via linear regression (slope * demand).
        #     What remains (the residual) reflects coordination ABOVE what
        #     demand alone would explain. This is the pure collusion signal.
        #
        #     Empirically: raw corr gap (colluding vs honest) = 0.006
        #                  residual corr gap                   = 0.026  (4× larger)
        residuals = np.zeros_like(price_matrix, dtype=np.float64)
        for i in range(n_drivers):
            p = price_matrix[i].astype(np.float64)
            d = demand_matrix[i].astype(np.float64)
            if np.std(d) > eps:
                slope       = np.cov(p, d)[0, 1] / (np.var(d) + eps)
                residuals[i] = p - slope * d
            else:
                residuals[i] = p - p.mean()

        pairwise_corrs = []
        for i in range(n_drivers):
            for j in range(i + 1, n_drivers):
                ri, rj = residuals[i], residuals[j]
                if np.std(ri) > eps and np.std(rj) > eps:
                    corr = float(np.corrcoef(ri, rj)[0, 1])
                    pairwise_corrs.append(corr)

        if pairwise_corrs:
            corr_mean = float(np.mean(pairwise_corrs))
            corr_std  = float(np.std(pairwise_corrs))
        else:
            corr_mean, corr_std = 0.0, 0.0

        # [10] Std of pairwise correlations
        #      LOW std = all pairs are equally correlated = uniform synchrony = suspicious
        #      HIGH std = some pairs correlated, some not = natural variation

        # [11] Dispersion trend — is price dispersion decreasing over the window?
        #      Fit a linear trend to per-timestep dispersion values
        #      Negative slope = prices converging = suspicious
        dispersions = window_features[4, :]   # row 4 = coefficient of variation
        if W > 1:
            x = np.arange(W, dtype=np.float32)
            # Linear regression slope via formula: slope = cov(x,y) / var(x)
            x_mean  = x.mean()
            d_mean  = dispersions.mean()
            slope   = float(np.sum((x - x_mean) * (dispersions - d_mean)) / (np.sum((x - x_mean)**2) + eps))
        else:
            slope = 0.0

        # Normalize slope to [-1, 1] range (clip extremes)
        disp_trend = float(np.clip(slope * 10, -1.0, 1.0))

        # [12] Offline synchronization score
        #      How synchronized are offline events across drivers in the window?
        #      Measure: at each timestep, std of online status across drivers
        #      LOW std at most timesteps = everyone online or offline together = suspicious
        online_std_per_step = np.std(online_matrix, axis=0)   # (W,)
        # Score: fraction of timesteps where at least one driver is offline AND they're synchronized
        any_offline    = (online_matrix.mean(axis=0) < 1.0)
        synced_offline = (online_std_per_step < 0.3) & any_offline
        offline_sync   = float(np.mean(synced_offline))

        # Write window-level features (broadcast across time axis)
        result = window_features.copy()
        result[9,  :] = corr_mean
        result[10, :] = corr_std
        result[11, :] = disp_trend
        result[12, :] = offline_sync

        return result

    # ------------------------------------------------------------------
    # Feature name lookup (useful for ablation studies)
    # ------------------------------------------------------------------

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # Per-driver (aggregated)
            "mean_price_norm",
            "online_fraction",
            "mean_demand_norm",
            "mean_acceptance_rate",
            # Cross-driver (per timestep)
            "price_cv",                    # coefficient of variation (std/mean)
            "group_mean_price_norm",
            "online_fraction_cross",
            "direction_agreement",
            "demand_weighted_dispersion",
            # Window-level
            "pairwise_corr_mean",
            "pairwise_corr_std",
            "dispersion_trend",
            "offline_sync_score",
        ]