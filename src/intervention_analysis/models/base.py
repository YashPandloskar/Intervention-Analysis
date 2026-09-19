"""Shared containers passed between the pipeline and the individual models."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class Problem:
    """One train/test split of a price series, optionally with intervention variables D_t."""

    y_train: np.ndarray
    y_test: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    d_train: np.ndarray | None = None  # (n_train, K) intervention variables
    d_test: np.ndarray | None = None  # (n_test, K)

    @property
    def n_interventions(self) -> int:
        return 0 if self.d_train is None else self.d_train.shape[1]

    def without_intervention(self) -> "Problem":
        return replace(self, d_train=None, d_test=None)


@dataclass
class Forecast:
    """Predictions for the test window.

    one_step:   y_t predicted from the true history up to t-1 (rolling, no refit).
    multi_step: y_t predicted from the end of training only, i.e. the whole test
                window is forecast in one go. Only meaningful for models that
                are recursive by construction (ARIMA / SARIMA).
    """

    one_step: np.ndarray
    multi_step: np.ndarray | None = None
