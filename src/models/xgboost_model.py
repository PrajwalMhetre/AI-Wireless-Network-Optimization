"""XGBoost model wrapper for later ML phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class XGBoostChannelModel:
    """Thin wrapper around XGBoost's classifier API."""

    random_state: int = 42
    n_estimators: int = 200
    model: Any | None = None

    def fit(self, features: Any, target: Any) -> "XGBoostChannelModel":
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.model.fit(features, target)
        return self

    def predict(self, features: Any) -> Any:
        if self.model is None:
            raise RuntimeError("XGBoostChannelModel must be fitted before predict")
        return self.model.predict(features)

