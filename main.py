"""Command-line interface for the Beat-the-Bookie pipeline.

Usage:

    python main.py train                          # full training + evaluation
    python main.py predict --input path/test.csv  # generate FINAL_PREDICTIONS.csv

Run ``python main.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data", help="Folder containing the *_games.csv files (default: data)")


def main(argv: list[str] | None = None) -> int:
    # Make the local ``src`` package importable when running from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

    parser = argparse.ArgumentParser(description="Beat the Bookie — Premier League match outcome prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train all models and report metrics")
    _add_common_args(p_train)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--bayes-iter", type=int, default=30)
    p_train.add_argument("--plots-dir", default="outputs/plots")
    p_train.add_argument("--quiet", action="store_true")

    p_pred = sub.add_parser("predict", help="Generate final test-set predictions")
    _add_common_args(p_pred)
    p_pred.add_argument("--input", default="data/sample-submission.csv", help="Path to the fixture CSV to predict")
    p_pred.add_argument("--output", default="FINAL_PREDICTIONS.csv")

    args = parser.parse_args(argv)

    from beat_the_bookie.pipeline import run_full_training, run_predictions

    if args.command == "train":
        run_full_training(
            data_dir=args.data_dir,
            epochs=args.epochs,
            bayes_iter=args.bayes_iter,
            plots_dir=args.plots_dir,
            verbose=not args.quiet,
        )
    elif args.command == "predict":
        run_predictions(
            test_data_path=args.input,
            data_dir=args.data_dir,
            output_path=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
