import numpy as np
import pandas as pd
import pytest

from intervention_analysis.changepoint import (
    detect_change_points,
    estimable_columns,
    make_intervention_matrix,
)
from intervention_analysis.metrics import regression_metrics
from intervention_analysis.models.base import Problem
from intervention_analysis.models.deep import make_windows
from intervention_analysis.models.statistical import fit_sarimax


@pytest.fixture
def step_series():
    """Noisy random walk with a clear level shift at position 150."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2024-01-01", periods=300)
    values = 100 + np.cumsum(rng.normal(0, 0.3, 300))
    values[150:] += 25
    return pd.Series(values, index=index, name="Close")


def test_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 5])
    assert m["MSE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(4 / 3))
    assert m["MAE"] == pytest.approx(2 / 3)


def test_change_point_found_near_true_shift(step_series):
    dates = detect_change_points(step_series, width=40, n_bkps=1)
    position = step_series.index.get_loc(dates[0])
    assert abs(position - 150) <= 5


def test_step_and_pulse_matrices(step_series):
    date = step_series.index[100]
    step = make_intervention_matrix(step_series.index, [date], "step")
    pulse = make_intervention_matrix(step_series.index, [date], "pulse")
    assert step["D1"].sum() == 200 and step["D1"].iloc[99] == 0 and step["D1"].iloc[100] == 1
    assert pulse["D1"].sum() == 1 and pulse["D1"].iloc[100] == 1
    with pytest.raises(ValueError):
        make_intervention_matrix(step_series.index, [date], "ramp")


def test_interventions_in_test_window_are_dropped(step_series):
    dates = [step_series.index[100], step_series.index[280]]
    matrix = make_intervention_matrix(step_series.index, dates)
    assert list(estimable_columns(matrix, n_train=240).columns) == ["D1"]


def test_windows_align_targets_and_interventions():
    y = np.arange(10, dtype=float)
    d = (np.arange(10) >= 6).astype(float).reshape(-1, 1)
    lags, d_seq, d_now, targets = make_windows(y, d, lookback=3)
    assert lags.shape == (7, 3) and d_seq.shape == (7, 3, 1) and d_now.shape == (7, 1)
    # Window i predicts y[i + 3] from y[i:i + 3] ...
    assert np.array_equal(lags[2], [2, 3, 4]) and targets[2] == 5 and y[targets[2]] == 5
    # ... and sees the intervention flags of the predicted days (i+1 .. i+3).
    assert np.array_equal(d_seq[3, :, 0], d[4:7, 0]) and d_now[3, 0] == d[6, 0]


def test_arima_intervention_uses_known_regime_in_test_window():
    rng = np.random.default_rng(1)
    n, n_train = 300, 240
    index = pd.bdate_range("2024-01-01", periods=n)
    # Regime indicator that is on twice in training and switches on again in the test window;
    # each time it is on the price sits 25 above its random-walk baseline.
    d = np.zeros((n, 1))
    d[100:150] = 1
    d[200:225] = 1
    d[260:] = 1
    y = 100 + np.cumsum(rng.normal(0, 0.3, n)) + 25 * d[:, 0]
    problem = Problem(y[:n_train], y[n_train:], index[:n_train], index[n_train:], d[:n_train], d[n_train:])

    with_d = fit_sarimax(problem, (1, 1, 0))[0]
    without = fit_sarimax(problem.without_intervention(), (1, 1, 0))[0]
    rmse = lambda f: np.sqrt(np.mean((problem.y_test - f) ** 2))  # noqa: E731
    # Only the model that knows D_t can anticipate the +25 jump at position 260.
    assert rmse(with_d.multi_step) < rmse(without.multi_step) / 2
    assert with_d.one_step.shape == problem.y_test.shape
