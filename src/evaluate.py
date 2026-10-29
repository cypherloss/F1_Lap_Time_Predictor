"""Evaluate the lap-time modeling workflow with race-grouped cross-validation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import CLEAN_PARQUET
from src.train import EvaluationMetrics, evaluate_model_cv


def format_report(metrics: EvaluationMetrics) -> str:
    """Format fold metrics and aggregate out-of-fold scores for console output."""
    lines = [
        f"Race-grouped cross-validation ({metrics['races']} races, {metrics['rows']} laps)",
        "",
        f"{'Fold':>4} {'Train races':>12} {'Valid races':>12} "
        f"{'MAE (s)':>10} {'RMSE (s)':>10} {'R2':>8}",
    ]
    for fold in metrics["folds"]:
        lines.append(
            f"{fold['fold']:>4} {fold['train_races']:>12} "
            f"{fold['validation_races']:>12} {fold['mae']:>10.3f} "
            f"{fold['rmse']:>10.3f} {fold['r2']:>8.3f}"
        )
    lines.extend([
        "",
        f"Out-of-fold MAE: {metrics['mae']:.3f} s",
        f"Out-of-fold RMSE: {metrics['rmse']:.3f} s",
        f"Out-of-fold R2: {metrics['r2']:.3f}",
        f"Baseline MAE: {metrics['baseline_mae']:.3f} s",
        f"Improvement over baseline: {metrics['improvement_pct']:.1f}%",
    ])
    return "\n".join(lines)


def main() -> None:
    """Load cleaned laps, run grouped evaluation, and optionally save JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=CLEAN_PARQUET,
        help=f"Clean lap parquet file (default: {CLEAN_PARQUET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving machine-readable evaluation metrics as JSON.",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Clean lap data not found: {args.data}. Run `python -m src.clean` first.")
    metrics = evaluate_model_cv(pd.read_parquet(args.data))
    print(format_report(metrics))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
