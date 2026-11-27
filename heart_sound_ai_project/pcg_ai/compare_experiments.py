"""Create an easy-to-read comparison table for saved training experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def _percent(value):
    return None if value is None else round(float(value) * 100, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved heart-sound AI experiment runs.")
    parser.add_argument("--root", type=Path, default=Path("experiments"), help="Folder containing experiment subfolders")
    args = parser.parse_args()
    root = args.root
    rows = []
    for details_path in sorted(root.glob("*/experiment_details.json")):
        run_dir = details_path.parent
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (run_dir / "metrics_summary.csv").exists():
            continue  # unfinished or failed run
        val = details.get("validation_metrics", {})
        test = details.get("held_out_test_metrics", {})
        split_counts = details.get("recording_counts_after_audio_processing", {})
        rows.append({
            "experiment_id": details.get("experiment_id", run_dir.name),
            "date_and_time": details.get("started_at_local", ""),
            "maximum_epochs_requested": details.get("maximum_epochs_requested", ""),
            "epochs_completed": details.get("epochs_completed", ""),
            "best_epoch": details.get("best_epoch_selected_by_validation_loss", ""),
            "random_seed": details.get("random_seed", ""),
            "training_recordings": split_counts.get("training", ""),
            "validation_recordings": split_counts.get("validation", ""),
            "test_recordings": split_counts.get("test", ""),
            "validation_accuracy_percent": _percent(val.get("accuracy")),
            "validation_balanced_accuracy_percent": _percent(val.get("balanced_accuracy")),
            "test_accuracy_percent_descriptive_only": _percent(test.get("accuracy")),
            "test_balanced_accuracy_percent_descriptive_only": _percent(test.get("balanced_accuracy")),
            "experiment_folder": str(run_dir),
        })
    if not rows:
        print("No completed experiment folders were found yet.")
        print(f"Expected folders like: {root}/YYYY-MM-DD_HH-MM-SS_seed42_max20epochs/")
        return
    frame = pd.DataFrame(rows).sort_values("date_and_time")
    output = root / "experiment_comparison.csv"
    frame.to_csv(output, index=False)
    print("\nEXPERIMENT COMPARISON")
    print("Validation metrics are the primary guide for choosing settings.")
    print("Test metrics are shown for transparency; avoid repeatedly tuning against the held-out test set.\n")
    display_cols = [
        "experiment_id", "maximum_epochs_requested", "epochs_completed", "best_epoch",
        "validation_accuracy_percent", "validation_balanced_accuracy_percent",
        "test_accuracy_percent_descriptive_only",
    ]
    print(frame[display_cols].to_string(index=False))
    print(f"\nSaved complete comparison to: {output.resolve()}")


if __name__ == "__main__":
    main()
