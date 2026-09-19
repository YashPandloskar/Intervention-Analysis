"""Classical time-series models with intervention variables: ARIMA, SARIMA, Prophet."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import Forecast, Problem

_NO_SEASONALITY = (0, 0, 0, 0)


def select_order(y: np.ndarray, d: int = 1, max_p: int = 3, max_q: int = 3) -> tuple[int, int, int]:
    """Choose the ARIMA(p, d, q) order with the lowest AIC on the training data."""
    best_aic, best_order = np.inf, (0, d, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    res = SARIMAX(
                        y, order=(p, d, q), trend="n",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                except Exception:  # noqa: BLE001 - a failed candidate is simply skipped
                    continue
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, (p, d, q)
    return best_order


def fit_sarimax(
    problem: Problem, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int] = _NO_SEASONALITY
) -> list[Forecast]:
    """ARIMA / SARIMA with the intervention variables entering as exogenous regressors.

    Without ``problem.d_train`` this is Eq. 1 (and the plain SARIMA equation); with it,
    the regression term delta * D_t of Eq. 2 (and gamma * D_t for SARIMA) is added.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            problem.y_train, exog=problem.d_train, order=order, seasonal_order=seasonal_order,
            trend="n", enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)

        multi = res.get_forecast(steps=len(problem.y_test), exog=problem.d_test).predicted_mean
        # Reuse the fitted parameters but let the filter see the true test observations,
        # giving genuine one-step-ahead predictions.
        extended = res.append(problem.y_test, exog=problem.d_test, refit=False)
        one = extended.get_prediction(start=len(problem.y_train)).predicted_mean
    return [Forecast(one_step=np.asarray(one), multi_step=np.asarray(multi))]


def fit_prophet(
    problem: Problem, country_holidays: str | None = "US", yearly_seasonality: bool | str = "auto"
) -> list[Forecast]:
    """Prophet: y = g + s + h + delta * I_t + e (Eq. 9 / 10).

    Prophet is not autoregressive, so its one-step and multi-step forecasts coincide.
    """
    from prophet import Prophet  # imported lazily: it is slow to import

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)

    columns = [f"D{k}" for k in range(1, problem.n_interventions + 1)]
    train = pd.DataFrame({"ds": problem.dates_train, "y": problem.y_train})
    test = pd.DataFrame({"ds": problem.dates_test})
    if columns:
        train[columns] = problem.d_train
        test[columns] = problem.d_test

    model = Prophet(daily_seasonality=False, yearly_seasonality=yearly_seasonality)
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    for column in columns:
        model.add_regressor(column, standardize=False)
    model.fit(train)

    yhat = model.predict(test)["yhat"].to_numpy()
    return [Forecast(one_step=yhat, multi_step=yhat)]
