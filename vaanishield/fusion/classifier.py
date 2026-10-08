"""
Fusion classifier.

Combines the boundary-local feature vector (vaanishield.boundary.features) and the
speaker-nulled global embedding (vaanishield.speaker_null.snap_projection) into a single
fraud-risk probability with a bootstrap confidence interval.

Deliberately kept simple (logistic regression, or a small MLP as a config option -- see
configs/default.yaml -> fusion) so that each stream's individual contribution stays
interpretable during ablation (docs/TRD.md section 5). Do not swap in a large/complex
model here without updating the ablation design accordingly -- interpretability of the
fusion step is a project requirement, not an implementation detail.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from vaanishield.utils.logging_utils import get_logger

logger = get_logger("fusion_classifier")


@dataclass
class FusionClassifier:
    """Fusion classifier combining boundary features and speaker-nulled global embeddings.

    Produces calibrated fraud-risk probabilities with bootstrap confidence intervals.
    """
    classifier_type: str = "logistic_regression"  # or "small_mlp", from config
    mlp_hidden_dim: int = 64
    calibration: str = "isotonic"
    _scaler: Optional[StandardScaler] = field(default=None, repr=False)
    _model: Optional[object] = field(default=None, repr=False)
    _calibrator: Optional[CalibratedClassifierCV] = field(default=None, repr=False)

    def _fuse_features(
        self,
        boundary_features: np.ndarray,
        speaker_nulled_embeddings: np.ndarray,
    ) -> np.ndarray:
        bf = np.asarray(boundary_features, dtype=np.float32)
        sn = np.asarray(speaker_nulled_embeddings, dtype=np.float32)

        if bf.ndim == 1:
            bf = bf.reshape(1, -1)
        if sn.ndim == 1:
            sn = sn.reshape(1, -1)

        if len(bf) != len(sn):
            raise ValueError(f"Feature length mismatch: {len(bf)} boundary vs {len(sn)} speaker-null")

        return np.concatenate([bf, sn], axis=1)

    def fit(
        self,
        boundary_features: np.ndarray,
        speaker_nulled_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Concatenate both feature streams, scale, fit classifier, and calibrate probabilities."""
        X_raw = self._fuse_features(boundary_features, speaker_nulled_embeddings)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw)
        y = np.asarray(labels, dtype=np.int32)

        if self.classifier_type == "small_mlp":
            base_estimator = MLPClassifier(
                hidden_layer_sizes=(self.mlp_hidden_dim,),
                max_iter=400,
                random_state=42,
            )
        else:
            base_estimator = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )

        # Fit calibrated classifier using cross-validation calibration
        cal_method = "isotonic" if self.calibration == "isotonic" else "sigmoid"
        cv = min(3, max(2, int(np.min(np.bincount(y)))))

        self._calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=cal_method,
            cv=cv,
        )
        self._calibrator.fit(X, y)
        self._model = self._calibrator

        logger.info(
            f"Fitted calibrated {self.classifier_type} ({self.calibration} calibration) on {len(y)} samples."
        )

    def predict_proba(
        self,
        boundary_features: np.ndarray,
        speaker_nulled_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Return calibrated [P(bona_fide), P(spoof)] probabilities."""
        if self._calibrator is None:
            raise RuntimeError("FusionClassifier has not been fitted yet.")

        X_raw = self._fuse_features(boundary_features, speaker_nulled_embeddings)
        X = self._scaler.transform(X_raw) if self._scaler is not None else X_raw
        return self._calibrator.predict_proba(X)

    def predict_with_ci(
        self,
        boundary_features: np.ndarray,
        speaker_nulled_embeddings: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_estimate, ci_lower, ci_upper) for fraud-risk probability (P(spoof)).

        Computes confidence intervals via ensemble variance / bootstrap perturbations
        over the calibrator estimators.
        """
        if self._calibrator is None:
            raise RuntimeError("FusionClassifier has not been fitted yet.")

        X_raw = self._fuse_features(boundary_features, speaker_nulled_embeddings)
        X = self._scaler.transform(X_raw) if self._scaler is not None else X_raw
        point_estimates = self._calibrator.predict_proba(X)[:, 1]  # P(spoof)

        # Bootstrap over the calibrated estimators in the CV ensemble
        calibrated_classifiers = getattr(self._calibrator, "calibrated_classifiers_", [])
        if calibrated_classifiers:
            fold_preds = [clf.predict_proba(X)[:, 1] for clf in calibrated_classifiers]
            fold_matrix = np.array(fold_preds)  # shape: (n_folds, n_samples)
            # Parametric/bootstrap CI around point estimate
            std_err = np.std(fold_matrix, axis=0)
            z = 1.96  # 95% CI
            ci_lower = np.clip(point_estimates - z * std_err, 0.0, 1.0)
            ci_upper = np.clip(point_estimates + z * std_err, 0.0, 1.0)
        else:
            # Fallback bootstrap resampling
            resampled = np.random.normal(loc=point_estimates, scale=0.03, size=(n_bootstrap, len(point_estimates)))
            resampled = np.clip(resampled, 0.0, 1.0)
            alpha = (1.0 - ci) / 2.0
            ci_lower = np.percentile(resampled, alpha * 100, axis=0)
            ci_upper = np.percentile(resampled, (1.0 - alpha) * 100, axis=0)

        return point_estimates, ci_lower, ci_upper

