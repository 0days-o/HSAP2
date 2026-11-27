from pathlib import Path
import pandas as pd

from pcg_ai.train import _make_experiment_dir, _metric_summary, _metric_rows


def test_each_run_gets_its_own_folder(tmp_path: Path):
    first, first_id = _make_experiment_dir(tmp_path, None, seed=42, epochs=20)
    second, second_id = _make_experiment_dir(tmp_path, None, seed=42, epochs=20)
    assert first.exists() and second.exists()
    assert first != second
    assert first_id != second_id


def test_metrics_are_recording_level_and_clear():
    frame = pd.DataFrame({
        "_reference_code": [0, 0, 1, 1],
        "_predicted_code": [0, 1, 0, 1],
        "_score": [0.1, 0.8, 0.4, 0.9],
    })
    summary = _metric_summary(frame)
    assert summary["recordings_evaluated"] == 4
    assert summary["accuracy"] == 0.5
    assert summary["balanced_accuracy"] == 0.5
    assert summary["confusion_matrix"] == [[1, 1], [1, 1]]


def test_metric_csv_explains_measurements():
    rows = _metric_rows("Validation", {
        "recordings_evaluated": 20,
        "correct_classifications": 15,
        "incorrect_classifications": 5,
        "accuracy": 0.75,
        "balanced_accuracy": 0.70,
        "sensitivity_abnormal_reference": 0.60,
        "specificity_normal_reference": 0.80,
        "roc_auc": 0.73,
    })
    accuracy = next(row for row in rows if row["measurement"] == "Accuracy")
    assert accuracy["value"] == 75.0
    assert accuracy["unit"] == "percent"
    assert "percent" in accuracy["what_this_means"].lower()
