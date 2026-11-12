"""TensorBoard event-file → CSV conversion utilities.

Extracted from analysis_v2.ipynb section 2 (TensorBoard Data Processing).
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Optional

import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError


def tensorboard_to_csv(event_file: str, csv_file: str) -> None:
    """Convert TensorBoard event file data to a single CSV (Step, Tag, Value)."""
    data_rows = []

    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.HasField("simple_value"):
                tag = v.tag
                value = v.simple_value
                step = e.step
                data_rows.append([step, tag, value])

    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Tag", "Value"])
        writer.writerows(data_rows)

    print(f"Data from {event_file} has been written to {csv_file}")


def tensorboard_to_separate_csv(event_file: str, output_dir: str) -> None:
    """Convert a TensorBoard event file into one CSV per scalar tag."""
    tag_data = defaultdict(list)

    try:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            try:
                for v in e.summary.value:
                    if v.HasField("simple_value"):
                        tag = v.tag
                        value = v.simple_value
                        step = e.step
                        tag_data[tag].append([step, value])
            except Exception:
                print(f"Skipped a corrupt record in file: {event_file}")
    except DataLossError:
        print(
            "Encountered DataLossError. Possibly due to incomplete writes in "
            f"file: {event_file}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for tag, data_rows in tag_data.items():
        filename = os.path.join(output_dir, f"{tag.replace('/', '_')}_data.csv")
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Value"])
            writer.writerows(data_rows)
        print(f"Data for tag '{tag}' has been written to {filename}")


def is_file_stable(file_path: str, wait_time: float = 1.0) -> bool:
    """Check if a file is stable (not being written to)."""
    initial_size = os.path.getsize(file_path)
    time.sleep(wait_time)
    final_size = os.path.getsize(file_path)
    return initial_size == final_size


def process_tensorboard_results(
    parent_dir: str,
    output_parent_dir: str,
    require_stable: bool = False,
    stable_wait: float = 1.0,
    run_filter: Optional[list[str]] = None,
) -> int:
    """Walk ``parent_dir`` and convert every ``tfevents`` file to per-tag CSVs.

    Directory structure under ``parent_dir`` is mirrored into ``output_parent_dir``.

    Args:
        parent_dir: Root containing run folders with TensorBoard event files.
        output_parent_dir: Root where CSV trees should be written (e.g. ``res/...``).
        require_stable: If True, skip files still being written.
        stable_wait: Seconds between size checks when ``require_stable`` is True.
        run_filter: Optional list of run folder basenames / relative paths to keep.
            If set, only matching run trees are converted.

    Returns:
        Number of event files successfully processed.
    """
    parent_dir = os.path.abspath(parent_dir)
    output_parent_dir = os.path.abspath(output_parent_dir)
    os.makedirs(output_parent_dir, exist_ok=True)

    filter_set = None
    if run_filter:
        filter_set = {os.path.normpath(x).rstrip("/") for x in run_filter}

    n_ok = 0
    for root, _dirs, files in os.walk(parent_dir):
        relative_path = os.path.relpath(root, parent_dir)
        if filter_set is not None:
            # Match either the relative path, its first component, or basename.
            top = relative_path.split(os.sep)[0] if relative_path != "." else ""
            candidates = {
                os.path.normpath(relative_path),
                os.path.normpath(top),
                os.path.basename(root),
            }
            if relative_path == ".":
                # Only filter at run-subdir level; still walk root.
                pass
            elif candidates.isdisjoint(filter_set) and not any(
                os.path.normpath(relative_path).startswith(f + os.sep)
                or os.path.normpath(relative_path) == f
                for f in filter_set
            ):
                continue

        for file in files:
            if "tfevents" not in file:
                continue
            event_file = os.path.join(root, file)
            if require_stable and not is_file_stable(event_file, wait_time=stable_wait):
                print(f"Skipping unstable file (still writing?): {event_file}")
                continue

            output_dir = os.path.join(output_parent_dir, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            try:
                tensorboard_to_separate_csv(event_file, output_dir)
                print(f"Processed {event_file} -> {output_dir}")
                n_ok += 1
            except Exception as e:
                print(f"Failed to process {event_file}: {e}")
    return n_ok
