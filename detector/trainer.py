"""
trainer.py — Training loop, evaluation, and ablation utilities.

Handles:
  - Training with early stopping
  - Per-mode evaluation (precision/recall/F1 per collusion type)
  - Ablation: cross-driver features only vs per-driver only vs both
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from detector.tcn import CollusionTCN
from features.extractor import PER_DRIVER_DIM, CROSS_DRIVER_DIM, WINDOW_LEVEL_DIM


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Controls the training loop.

    Attributes
    ----------
    epochs          : maximum training epochs
    batch_size      : samples per gradient step
    lr              : learning rate
    weight_decay    : L2 regularization strength
    patience        : early stopping — stop if val loss doesn't improve for N epochs
    pos_weight      : weight for positive class in BCE loss
                      set > 1 if dataset is imbalanced (more honest than collusion)
    hidden_dim      : TCN hidden channels
    dropout         : dropout rate
    seed            : for reproducibility
    """
    epochs:      int   = 60
    batch_size:  int   = 64
    lr:          float = 1e-3
    weight_decay:float = 1e-4
    patience:    int   = 10
    pos_weight:  float = 1.0   # set to n_neg/n_pos if imbalanced
    hidden_dim:  int   = 32
    dropout:     float = 0.2
    seed:        int   = 42


# ---------------------------------------------------------------------------
# Feature masking for ablation studies
# ---------------------------------------------------------------------------

# Feature index ranges (from extractor.py)
PER_DRIVER_INDICES   = list(range(0, PER_DRIVER_DIM))                                    # [0,1,2,3]
CROSS_DRIVER_INDICES = list(range(PER_DRIVER_DIM, PER_DRIVER_DIM + CROSS_DRIVER_DIM + WINDOW_LEVEL_DIM))  # [4..12]

def mask_features(X: np.ndarray, mode: str) -> np.ndarray:
    """
    Zero out features for ablation studies.

    mode options:
      'all'          — use all 13 features (default)
      'per_driver'   — zero out cross-driver features, keep per-driver only
      'cross_driver' — zero out per-driver features, keep cross-driver only
    """
    X = X.copy()
    if mode == 'per_driver':
        X[:, CROSS_DRIVER_INDICES, :] = 0.0
    elif mode == 'cross_driver':
        X[:, PER_DRIVER_INDICES, :] = 0.0
    # 'all' → no masking
    return X


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains a CollusionTCN and evaluates it.

    Usage:
        trainer = Trainer(config=TrainConfig())
        model, history = trainer.train(X_train, y_train, X_val, y_val)
        results = trainer.evaluate(model, X_test, y_test, meta_test)
    """

    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        verbose: bool = True,
        feature_mode: str = 'all',
    ) -> Tuple["CollusionTCN", Dict]:
        """
        Train the TCN with early stopping on validation loss.

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : validation data (for early stopping)
        verbose          : print progress
        feature_mode     : 'all' | 'per_driver' | 'cross_driver' (for ablation)

        Returns
        -------
        model   : best model (lowest val loss)
        history : dict of train_loss, val_loss, val_acc per epoch
        """
        cfg = self.config

        # Apply feature masking for ablation
        X_train = mask_features(X_train, feature_mode)
        X_val   = mask_features(X_val,   feature_mode)

        # Convert to tensors
        Xt = torch.FloatTensor(X_train)
        yt = torch.FloatTensor(y_train)
        Xv = torch.FloatTensor(X_val)
        yv = torch.FloatTensor(y_val)

        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size = cfg.batch_size,
            shuffle    = True,
        )

        # Build model
        n_features = X_train.shape[1]
        model = CollusionTCN(
            n_features = n_features,
            hidden_dim = cfg.hidden_dim,
            dropout    = cfg.dropout,
        )

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")
            print(f"Feature mode: {feature_mode}")
            print(f"Training on {len(y_train)} windows, validating on {len(y_val)}\n")

        # Loss: weighted BCE to handle class imbalance
        # pos_weight > 1 = penalize missing collusion more than false positives
        pos_weight = torch.tensor([cfg.pos_weight])
        criterion  = nn.BCELoss()   # sigmoid already applied in model
        optimizer  = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        best_val_loss  = float('inf')
        best_state     = None
        patience_count = 0

        for epoch in range(cfg.epochs):
            # --- Training ---
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_preds = model(Xv)
                val_loss  = criterion(val_preds, yv).item()
                val_labels = (val_preds > 0.5).float()
                val_acc    = (val_labels == yv).float().mean().item()
                val_f1     = self._f1_score(
                    yv.numpy(), val_labels.numpy()
                )

            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{cfg.epochs} | "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_acc={val_acc:.3f} | "
                      f"val_F1={val_f1:.3f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss  = val_loss
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    if verbose:
                        print(f"\n  Early stopping at epoch {epoch+1} "
                              f"(no improvement for {cfg.patience} epochs)")
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose:
            print(f"\n  Best val loss: {best_val_loss:.4f}")

        return model, history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model:    "CollusionTCN",
        X_test:   np.ndarray,
        y_test:   np.ndarray,
        meta_test: List[Dict],
        threshold: float = 0.5,
        feature_mode: str = 'all',
    ) -> Dict:
        """
        Full evaluation: overall metrics + per-mode breakdown.

        Returns dict with:
          overall: precision, recall, F1, accuracy, AUC
          per_mode: same metrics broken down by collusion mode
          predictions: raw probabilities for each test window
        """
        X_test = mask_features(X_test, feature_mode)
        model.eval()

        with torch.no_grad():
            probs  = model(torch.FloatTensor(X_test)).numpy()
            preds  = (probs > threshold).astype(float)

        results = {
            "overall":     self._compute_metrics(y_test, preds, probs),
            "per_mode":    {},
            "predictions": probs,
            "threshold":   threshold,
        }

        # Per-mode breakdown
        modes = set(m["mode"] for m in meta_test)
        for mode in modes:
            idx = [i for i, m in enumerate(meta_test) if m["mode"] == mode]
            if idx:
                results["per_mode"][mode] = self._compute_metrics(
                    y_test[idx], preds[idx], probs[idx]
                )

        return results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs:  np.ndarray,
    ) -> Dict:
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        tn = float(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        fpr       = fp / (fp + tn + 1e-8)   # false positive rate

        # AUC via trapezoidal rule
        auc = self._auc(y_true, probs)

        return {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "accuracy":  round(accuracy,  4),
            "fpr":       round(fpr,       4),   # false positive rate
            "auc":       round(auc,       4),
            "n":         int(len(y_true)),
            "n_pos":     int(y_true.sum()),
        }

    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        return 2 * p * r / (p + r + 1e-8)

    def _auc(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        """Compute AUC-ROC via trapezoidal rule."""
        thresholds = np.linspace(0, 1, 100)
        tprs, fprs = [], []
        for t in thresholds:
            preds = (probs >= t).astype(float)
            tp = float(((preds == 1) & (y_true == 1)).sum())
            fp = float(((preds == 1) & (y_true == 0)).sum())
            fn = float(((preds == 0) & (y_true == 1)).sum())
            tn = float(((preds == 0) & (y_true == 0)).sum())
            tprs.append(tp / (tp + fn + 1e-8))
            fprs.append(fp / (fp + tn + 1e-8))
        # Sort by FPR for trapz
        pairs = sorted(zip(fprs, tprs))
        fprs_s = [p[0] for p in pairs]
        tprs_s = [p[1] for p in pairs]
        return float(np.trapezoid(tprs_s, fprs_s))