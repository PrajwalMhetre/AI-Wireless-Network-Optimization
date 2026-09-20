"""Random Forest model wrapper for later ML phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RandomForestChannelModel:
    """Thin wrapper around scikit-learn's RandomForestClassifier."""

    random_state: int = 42
    n_estimators: int = 200
    model: Any | None = None

    def fit(self, features: Any, target: Any) -> "RandomForestChannelModel":
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(features, target)
        return self

    def predict(self, features: Any) -> Any:
        if self.model is None:
            raise RuntimeError("RandomForestChannelModel must be fitted before predict")
        return self.model.predict(features)

