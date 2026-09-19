# Results

## Paper protocol (time-series models: multi-step forecast of the test window; deep models: one-step-ahead)

### TSLA - Without intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 4265.780 | 65.313 | 59.512 |
| Time Series Analysis | ARIMA | 1386.113 | 37.231 | 31.211 |
| Time Series Analysis | SARIMA | 1390.765 | 37.293 | 31.282 |
| Deep Learning | RNN | 184.043 | 13.564 | 10.285 |
| Deep Learning | **MLP** | 159.818 | 12.641 | 9.644 |
| Deep Learning | LSTM | 161.071 | 12.691 | 9.669 |

### TSLA - With intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 1546.896 | 39.331 | 34.221 |
| Time Series Analysis | ARIMA | 1377.978 | 37.121 | 31.098 |
| Time Series Analysis | SARIMA | 1379.044 | 37.135 | 31.125 |
| Deep Learning | RNN | 230.062 | 15.090 | 11.897 |
| Deep Learning | MLP | 205.301 | 14.314 | 11.474 |
| Deep Learning | **LSTM** | 203.867 | 14.259 | 11.441 |

### IBM - Without intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 4717.159 | 68.682 | 59.958 |
| Time Series Analysis | ARIMA | 946.538 | 30.766 | 20.911 |
| Time Series Analysis | SARIMA | 933.016 | 30.545 | 20.799 |
| Deep Learning | RNN | 130.485 | 11.423 | 6.942 |
| Deep Learning | MLP | 135.915 | 11.656 | 6.865 |
| Deep Learning | **LSTM** | 118.131 | 10.869 | 6.369 |

### IBM - With intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 846.070 | 29.087 | 21.233 |
| Time Series Analysis | ARIMA | 908.815 | 30.147 | 20.697 |
| Time Series Analysis | SARIMA | 873.687 | 29.558 | 20.568 |
| Deep Learning | RNN | 231.521 | 15.212 | 11.544 |
| Deep Learning | **MLP** | 150.378 | 12.259 | 7.530 |
| Deep Learning | LSTM | 234.464 | 15.296 | 11.858 |

## Like-for-like protocol (every model: one-step-ahead)

### TSLA - Without intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 4265.780 | 65.313 | 59.512 |
| Time Series Analysis | ARIMA | 241.365 | 15.536 | 12.043 |
| Time Series Analysis | SARIMA | 241.323 | 15.535 | 12.040 |
| Deep Learning | RNN | 184.043 | 13.564 | 10.285 |
| Deep Learning | **MLP** | 159.818 | 12.641 | 9.644 |
| Deep Learning | LSTM | 161.071 | 12.691 | 9.669 |

### TSLA - With intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 1546.896 | 39.331 | 34.221 |
| Time Series Analysis | **ARIMA** | 153.341 | 12.383 | 9.271 |
| Time Series Analysis | SARIMA | 156.242 | 12.500 | 9.385 |
| Deep Learning | RNN | 230.062 | 15.090 | 11.897 |
| Deep Learning | MLP | 205.301 | 14.314 | 11.474 |
| Deep Learning | LSTM | 203.867 | 14.259 | 11.441 |

### IBM - Without intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 4717.159 | 68.682 | 59.958 |
| Time Series Analysis | ARIMA | 114.156 | 10.684 | 6.202 |
| Time Series Analysis | **SARIMA** | 114.043 | 10.679 | 6.224 |
| Deep Learning | RNN | 130.485 | 11.423 | 6.942 |
| Deep Learning | MLP | 135.915 | 11.656 | 6.865 |
| Deep Learning | LSTM | 118.131 | 10.869 | 6.369 |

### IBM - With intervention

| Family | Model | MSE | RMSE | MAE |
|---|---|---:|---:|---:|
| Time Series Analysis | PROPHET | 846.070 | 29.087 | 21.233 |
| Time Series Analysis | ARIMA | 117.383 | 10.834 | 6.306 |
| Time Series Analysis | **SARIMA** | 115.941 | 10.768 | 6.258 |
| Deep Learning | RNN | 231.521 | 15.212 | 11.544 |
| Deep Learning | MLP | 150.378 | 12.259 | 7.530 |
| Deep Learning | LSTM | 234.464 | 15.296 | 11.858 |
