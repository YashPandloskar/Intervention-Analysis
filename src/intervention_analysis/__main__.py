"""Command line entry point:  python -m intervention_analysis --config config.yaml"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from .pipeline import run_ticker, write_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", type=Path, help="path to the YAML configuration")
    parser.add_argument("--tickers", nargs="+", help="override the tickers in the configuration")
    parser.add_argument("--refresh", action="store_true", help="re-download prices instead of using the cache")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_root = Path(cfg["output"])

    all_metrics = [
        run_ticker(ticker, cfg, out_root / ticker, refresh=args.refresh)
        for ticker in args.tickers or cfg["tickers"]
    ]
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(out_root / "metrics.csv", index=False)
    write_summary(metrics, out_root / "summary.md")
    logging.info("Done. Results written to %s", out_root.resolve())


if __name__ == "__main__":
    main()
