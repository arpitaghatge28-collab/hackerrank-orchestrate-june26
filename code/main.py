"""
Entry point.

Usage:
    python main.py                          # eval + predict
    python main.py --mode eval              # only evaluate sample
    python main.py --mode predict           # only produce output.csv
    python main.py --dataset-dir ./dataset  # custom dataset path
"""

import argparse
import logging
import os
import sys

from src.pipeline import process_claims
from src.evaluate import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Claim Verification")
    parser.add_argument("--mode", choices=["eval", "predict", "both"], default="both")
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()

    d = args.dataset_dir
    required = [
        os.path.join(d, "claims.csv"),
        os.path.join(d, "sample_claims.csv"),
        os.path.join(d, "user_history.csv"),
        os.path.join(d, "evidence_requirements.csv"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        log.error("Missing files: %s", missing)
        sys.exit(1)

    if args.mode in ("eval", "both"):
        log.info("=== EVALUATION on sample_claims.csv ===")
        process_claims(
            claims_path       = os.path.join(d, "sample_claims.csv"),
            user_history_path = os.path.join(d, "user_history.csv"),
            evidence_req_path = os.path.join(d, "evidence_requirements.csv"),
            output_path       = "evaluation/sample_predictions.csv",
            base_dir          = d,
        )
        evaluate_predictions(
            predictions_path  = "evaluation/sample_predictions.csv",
            ground_truth_path = os.path.join(d, "sample_claims.csv"),
            report_path       = "evaluation/evaluation_report.md",
        )

    if args.mode in ("predict", "both"):
        log.info("=== PREDICTION on claims.csv ===")
        process_claims(
            claims_path       = os.path.join(d, "claims.csv"),
            user_history_path = os.path.join(d, "user_history.csv"),
            evidence_req_path = os.path.join(d, "evidence_requirements.csv"),
            output_path       = "output.csv",
            base_dir          = d,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()