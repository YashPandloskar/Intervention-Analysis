"""Deep-learning models with intervention awareness: RNN, LSTM and MLP (Keras).

How the intervention variable D_t enters each network:

* RNN / LSTM - D_t is concatenated to the input at every time step. Because the
  cell computes ``W . [x_t, D_t] = W_x . x_t + W_d . D_t``, this is exactly the extra
  ``delta * I_t`` (RNN, Eq. 13) and ``W . D_t`` (LSTM gates, Eq. 6-8) term of the paper.
* MLP - D_t is added to the pre-activation of the second layer,
  ``z2 = sigma(W2 . z1 + delta . D_t + b2)``, as in the paper.

Alignment: a window of ``lookback`` past prices predicts the price at time t, and
the intervention variables handed to the network are those of the *predicted* days
(t-lookback+1 .. t), mirroring how the exogenous regressor is used by ARIMA/SARIMA.
This treats the intervention as a known event, not something to be forecast.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

from .base import Forecast, Problem


def make_windows(y: np.ndarray, d: np.ndarray | None, lookback: int):
    """Slice a (scaled) series into supervised windows.

    Returns ``lags`` (N, L), ``d_seq`` (N, L, K) or None, ``d_now`` (N, K) or None and
    ``targets``, the index into ``y`` of the value each window predicts (L .. n-1).
    """
    n = len(y)
    lags = sliding_window_view(y, lookback)[: n - lookback]
    targets = np.arange(lookback, n)
    d_seq = d_now = None
    if d is not None:
        d_windows = sliding_window_view(d, lookback, axis=0).transpose(0, 2, 1)  # (n-L+1, L, K)
        d_seq = d_windows[1 : n - lookback + 1]
        d_now = d[lookback:]
    return lags, d_seq, d_now, targets


def _build_recurrent(kind: str, lookback: int, n_features: int, units: int, lr: float):
    from keras import Input, Model, layers, optimizers

    inputs = Input((lookback, n_features))
    cell = layers.SimpleRNN(units, activation="tanh") if kind == "rnn" else layers.LSTM(units)
    outputs = layers.Dense(1)(cell(inputs))
    model = Model(inputs, outputs)
    model.compile(optimizers.Adam(lr), loss="mse")
    return model


def _build_mlp(lookback: int, n_interventions: int, hidden: list[int], activation: str, lr: float):
    from keras import Input, Model, layers, optimizers

    lags = Input((lookback,))
    z1 = layers.Dense(hidden[0], activation=activation)(lags)
    if n_interventions:
        d_in = Input((n_interventions,))
        pre = layers.Add()([layers.Dense(hidden[1])(z1), layers.Dense(hidden[1], use_bias=False)(d_in)])
        z2 = layers.Activation(activation)(pre)
        inputs = [lags, d_in]
    else:
        z2 = layers.Dense(hidden[1], activation=activation)(z1)
        inputs = lags
    # Linear output: a sigmoid head would cap forecasts at the training maximum.
    outputs = layers.Dense(1)(z2)
    model = Model(inputs, outputs)
    model.compile(optimizers.Adam(lr), loss="mse")
    return model


def run_deep(kind: str, problem: Problem, cfg: dict) -> list[Forecast]:
    """Train ``kind`` (rnn | lstm | mlp) once per seed and return one forecast per seed."""
    import keras
    from keras import callbacks

    lookback = cfg["lookback"]
    n_train = len(problem.y_train)
    k = problem.n_interventions

    scaler = MinMaxScaler().fit(problem.y_train.reshape(-1, 1))  # fitted on training data only
    y_all = scaler.transform(np.concatenate([problem.y_train, problem.y_test]).reshape(-1, 1)).ravel()
    d_all = np.vstack([problem.d_train, problem.d_test]) if k else None
    lags, d_seq, d_now, targets = make_windows(y_all, d_all, lookback)

    if kind == "mlp":
        features = [lags] + ([d_now] if k else [])
    else:
        seq = lags[..., None]
        features = [np.concatenate([seq, d_seq], axis=-1) if k else seq]

    is_train = targets < n_train
    train_idx = np.flatnonzero(is_train)
    test_idx = np.flatnonzero(~is_train)
    # Early stopping is optional: it needs a chronological hold-out from the end of training,
    # which can sit in a different price regime and stop training too early.
    use_val = cfg["patience"] is not None
    n_val = max(1, int(len(train_idx) * cfg["val_frac"])) if use_val else 0
    fit_idx, val_idx = (train_idx[:-n_val], train_idx[-n_val:]) if use_val else (train_idx, None)

    def take(idx):
        chosen = [f[idx] for f in features]
        return chosen[0] if len(chosen) == 1 else chosen

    forecasts = []
    for seed in cfg["seeds"]:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        if kind == "mlp":
            model = _build_mlp(lookback, k, cfg["mlp_hidden"], cfg["mlp_activation"], cfg["learning_rate"])
        else:
            units = cfg["rnn_units"] if kind == "rnn" else cfg["lstm_units"]
            model = _build_recurrent(kind, lookback, features[0].shape[-1], units, cfg["learning_rate"])

        fit_kwargs = {}
        if use_val:
            fit_kwargs = {
                "validation_data": (take(val_idx), y_all[targets[val_idx]]),
                "callbacks": [callbacks.EarlyStopping(patience=cfg["patience"], restore_best_weights=True)],
            }
        model.fit(
            take(fit_idx), y_all[targets[fit_idx]],
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], verbose=0, **fit_kwargs,
        )
        pred = model(take(test_idx), training=False).numpy()
        forecasts.append(Forecast(one_step=scaler.inverse_transform(pred).ravel()))
    return forecasts
