#!/usr/bin/env python3
"""Convert TensorBoard event files to per-metric CSVs.

Extracted from analysis_v2.ipynb (section 2 / example cell 27).

Example (from this directory):

    python convert_tensorboard_to_csv.py \\
        --parent-dir ../runs_bandit_study2_v3_twoArms \\
        --output-dir res/runs_bandit_study2_v3_twoArms

    # Only convert specific run folders:
    python convert_tensorboard_to_csv.py \\
        --parent-dir ../runs_bandit_study2_v3_twoArms \\
        --output-dir res/runs_bandit_study2_v3_twoArms \\
        --runs RUN_A RUN_B
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tensorboard_utils import (
    process_tensorboard_results,
    tensorboard_to_csv,
    tensorboard_to_separate_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert TensorBoard tfevents files to CSV (notebook section 2)."
    )
    p.add_argument(
        "--analysis-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Working directory for relative paths (default: this script's dir).",
    )
    p.add_argument(
        "--parent-dir",
        type=str,
        required=True,
        help="Root directory containing run folders with tfevents files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for CSV output (mirrors parent-dir structure).",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional subset of run folder names / relative paths to convert.",
    )
    p.add_argument(
        "--require-stable",
        action="store_true",
        help="Skip event files that appear still to be written.",
    )
    p.add_argument(
        "--stable-wait",
        type=float,
        default=1.0,
        help="Seconds between size checks when --require-stable is set.",
    )
    p.add_argument(
        "--single-event",
        type=str,
        default=None,
        help="Convert one event file only (with --single-mode).",
    )
    p.add_argument(
        "--single-mode",
        choices=["separate", "combined"],
        default="separate",
        help="When --single-event is set: per-tag CSVs or one combined CSV.",
    )
    p.add_argument(
        "--single-csv",
        type=str,
        default=None,
        help="Output CSV path for --single-mode combined.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root).resolve()
    os.chdir(analysis_root)

    if args.single_event:
        event_file = Path(args.single_event)
        if not event_file.is_absolute():
            event_file = analysis_root / event_file
        event_file = event_file.resolve()
        if not event_file.is_file():
            raise FileNotFoundError(f"Event file not found: {event_file}")

        if args.single_mode == "combined":
            out = args.single_csv or str(
                Path(args.output_dir) / f"{event_file.name}.csv"
            )
            if not Path(out).is_absolute():
                out = str(analysis_root / out)
            tensorboard_to_csv(str(event_file), out)
        else:
            out_dir = Path(args.output_dir)
            if not out_dir.is_absolute():
                out_dir = analysis_root / out_dir
            tensorboard_to_separate_csv(str(event_file), str(out_dir.resolve()))
        return

    parent_dir = Path(args.parent_dir)
    if not parent_dir.is_absolute():
        parent_dir = analysis_root / parent_dir
    parent_dir = parent_dir.resolve()
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"Parent dir not found: {parent_dir}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = analysis_root / output_dir
    output_dir = output_dir.resolve()

    print(f"parent-dir: {parent_dir}")
    print(f"output-dir: {output_dir}")
    if args.runs:
        print(f"runs filter ({len(args.runs)}):")
        for r in args.runs:
            print(f"  - {r}")

    n = process_tensorboard_results(
        parent_dir=str(parent_dir),
        output_parent_dir=str(output_dir),
        require_stable=args.require_stable,
        stable_wait=args.stable_wait,
        run_filter=args.runs,
    )
    print(f"\nDone. Processed {n} event file(s).")


if __name__ == "__main__":
    main()
