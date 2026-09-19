"""Intervention analysis for stock price forecasting (time series + deep learning)."""
import os

# Keep TensorFlow's C++ start-up chatter out of the console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

__version__ = "0.1.0"
