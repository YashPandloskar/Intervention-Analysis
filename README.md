# Intervention Analysis for Stock Market Prediction

Implementation of the paper **"Intervention Analysis for Stock Market Prediction using Time Series Analysis and
Deep Learning Algorithms"** (Valecha, Pandloskar, Dadheech, Saval — Dwarkadas J. Sanghvi College of Engineering),
re-run on the latest available data (Tesla and IBM, two years ending 2026-09-18).

The idea: detect the dates on which a stock's behaviour shifts (an *intervention*), encode them as an
intervention variable `D_t`, and feed `D_t` into six forecasting models — ARIMA, SARIMA, Prophet, RNN, LSTM and
MLP — to see whether, and for which model, it improves prediction.

## Pipeline

| Paper section | What happens | Code |
|---|---|---|
| III-A Data | Daily OHLCV + Adj Close from Yahoo Finance, chronological 80 / 20 train-test split | [`data.py`](src/intervention_analysis/data.py) |
| III-D Intervention analysis | Sliding-window change-point search on an RBF-kernel fit term (Eq. 15) plus penalty term, `C_k = F_k + P_k` | [`changepoint.py`](src/intervention_analysis/changepoint.py) |
| III-B 1, 4 | ARIMA / SARIMA with `D_t` as exogenous regressor (Eq. 1-2) | [`models/statistical.py`](src/intervention_analysis/models/statistical.py) |
| III-B 3 | Prophet with `D_t` as an extra regressor and holiday term (Eq. 9-10) | [`models/statistical.py`](src/intervention_analysis/models/statistical.py) |
| III-B 2, 5, 6 | LSTM, RNN, MLP with `D_t` added to gates / hidden state / second layer (Eq. 3-8, 11-14, MLP) | [`models/deep.py`](src/intervention_analysis/models/deep.py) |
| III-C Evaluation | MSE, RMSE, MAE; true-vs-predicted plots | [`metrics.py`](src/intervention_analysis/metrics.py), [`plots.py`](src/intervention_analysis/plots.py) |

Every model is fitted twice on the same split: **without** intervention terms (the paper's "before intervention"
equations) and **with** them (the "after intervention" equations).

## Quick start

Python 3.10+ (developed on 3.11).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m intervention_analysis --config config.yaml   # or: intervention-analysis
pytest                                                  # unit tests (a few seconds)
```

A full run (2 tickers x 6 models x 2 variants, 3 seeds per network) takes roughly 10 minutes on CPU. Prices are
cached in [`data/`](data), so re-runs are reproducible and offline; add `--refresh` to download the latest prices,
and `--tickers AAPL MSFT` to try other companies. All settings live in [`config.yaml`](config.yaml).

Outputs land in `results/`:

```
results/summary.md            paper-style MSE / RMSE / MAE tables
results/metrics.csv           all metrics (deep models also report std over seeds)
results/<TICKER>/interventions.csv|png   detected intervention dates
results/<TICKER>/forecast_{baseline,intervention}.png   true vs predicted, all six models
results/<TICKER>/predictions.csv
```

## Results (data: 2024-09-19 to 2026-09-18, 501 trading days; 400 train / 101 test)

Detected interventions (`results/<TICKER>/interventions.csv`; the `event` column is left blank for you to
annotate with the news behind each date, as the paper does):

* **TSLA** — 2024-11-07, 2025-02-25, 2025-05-12, 2025-09-12, 2026-01-29, 2026-05-04
* **IBM** — 2025-01-30, 2025-07-24, 2025-09-18, 2026-02-11, 2026-07-14

An intervention inside the test window (TSLA 2026-05-04, IBM 2026-07-14) cannot be learned from training data, so it
is reported but not used as a regressor.

RMSE without -> with intervention terms (lower is better; full tables in [`results/summary.md`](results/summary.md)):

**Paper protocol** (time-series models forecast the whole 101-day test window from the end of training; deep
models predict one step ahead from a window of true past prices)

| Model | TSLA | IBM |
|---|---|---|
| Prophet | 65.31 -> **39.33** | 68.68 -> **29.09** |
| ARIMA | 37.23 -> 37.12 | 30.77 -> 30.15 |
| SARIMA | 37.29 -> 37.14 | 30.55 -> 29.56 |
| RNN | 13.56 -> 15.09 | 11.42 -> 15.21 |
| MLP | 12.64 -> 14.31 | 11.66 -> 12.26 |
| LSTM | 12.69 -> 14.26 | 10.87 -> 15.30 |

**Like-for-like protocol** (every model predicts one step ahead; only ARIMA / SARIMA change, Prophet is not
autoregressive and the deep models are identical to the table above)

| Model | TSLA | IBM |
|---|---|---|
| ARIMA | 15.54 -> **12.38** | 10.68 -> 10.83 |
| SARIMA | 15.54 -> **12.50** | 10.68 -> 10.77 |

![TSLA interventions](results/TSLA/interventions.png)
![IBM forecasts with intervention](results/IBM/forecast_intervention.png)

### What the results say

* Intervention terms help the **time-series** models — a lot for Prophet, whose trend otherwise extrapolates
  straight through regime shifts — and TSLA's one-step ARIMA/SARIMA. They do **not** help the deep models here;
  with ~400 training points the extra inputs mostly add variance.
* The paper's headline (**RNN is best**) is **not reproduced** on this data: the best deep model differs by
  ticker and variant (MLP / LSTM / RNN are within a few tenths of each other for the baseline).
* The paper's ~10x gap between deep and time-series models mostly disappears under the like-for-like protocol:
  a one-step-ahead ARIMA is on par with, or better than, the deep networks. In the paper's setup the deep models
  get true past prices at every step while ARIMA / SARIMA forecast ~100 days blind, which is not a fair comparison
  of model quality. Both protocols are reported so the paper's setup stays reproducible.
* Sanity check: simply predicting "yesterday's close" gives a one-step RMSE of **12.41 (TSLA)** and **10.58 (IBM)**.
  No model here clearly beats that, which is what a near random-walk price series looks like. There is also a
  single train/test split per ticker, so treat differences of a few tenths of RMSE as noise. This is research
  code, not investment advice.

## Modelling choices and deviations from the paper

The paper leaves several details open; these are the choices made here (all configurable).

* **Intervention variable.** A step dummy `D_t = 1` from each detected date on (`kind: pulse` gives a one-day
  shock). The paper detects several points but writes a single `D_t`, so each intervention gets its own column
  and `delta` is a vector.
* **Detection.** `ruptures.Window` with the RBF cost; the number of interventions is controlled by the penalty
  (`pen`) or fixed with `n_bkps`. Detection runs on the whole series, as in the paper; only interventions
  inside the training window are used as regressors, so the test period cannot leak in beyond the window edge.
  The intervention is treated as a known event at forecast time (its future value is given to the model).
* **RNN / LSTM.** `D_t` is concatenated to the input of every step, which is algebraically the extra `W . D_t` /
  `delta * I_t` term in the paper's equations. **MLP:** `D_t` is added to the pre-activation of layer 2. The MLP
  output layer is linear (a sigmoid head, as in the paper's equation, would cap forecasts at the training
  maximum) and the paper's "Q-network" wording is interpreted as a plain MLP.
* **Orders.** ARIMA `(p, 1, q)` is chosen by AIC on the training set and shared with SARIMA
  (seasonal `(1,0,1,5)`, weekly). The constant `c` is omitted because of the differencing.
* **Training.** Prices are min-max scaled on the training set; 20-day look-back; 300 epochs, Adam, no early
  stopping; each network is trained with 3 seeds and metrics are averaged. Early stopping on the last 10 % of the
  training window (available via `deep.patience`) was tried first and stopped the MLP / LSTM before they had fit
  the data, so it is off by default. That choice was made after looking at IBM test errors, so IBM numbers are
  slightly optimistic for the deep models.
* **Data.** The latest two years (the paper used ~2 years of IBM and Tesla data). `Close` is the target.

## Project layout

```
config.yaml            all settings
data/                  cached prices (TSLA.csv, IBM.csv)
results/               tables and figures produced by the last run
src/intervention_analysis/
    data.py  changepoint.py  metrics.py  pipeline.py  plots.py  __main__.py
    models/  base.py  statistical.py  deep.py
tests/                 unit tests
```

## Reference

K. Valecha, Y. Pandloskar, S. Dadheech, P. Saval, "Intervention Analysis for Stock Market Prediction using Times
Series Analysis and Deep Learning Algorithms", Dwarkadas J. Sanghvi College of Engineering, Mumbai.
