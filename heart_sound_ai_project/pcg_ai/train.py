"""Train an educational heart-sound CNN and preserve each run separately.

The split is made by recording before segment extraction, so windows from one
recording cannot cross train/validation/test subsets. The source files do not
consistently expose patient IDs; this is not guaranteed to be patient-independent.

All outputs are written into a new timestamped experiment folder. The test split
is reported for transparency, but use validation metrics for model selection.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .audio import INPUT_SHAPE, wav_to_spectrograms
from .dataset import Record, load_records
from .model import build_model


def _make_experiment_dir(root: Path, run_name: str | None, seed: int, epochs: int) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", run_name).strip("_-")[:40] or "experiment"
        experiment_id = f"{stamp}_{safe_name}_seed{seed}"
    else:
        experiment_id = f"{stamp}_seed{seed}_max{epochs}epochs"
    candidate = root / experiment_id
    suffix = 2
    while candidate.exists():
        candidate = root / f"{experiment_id}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate, candidate.name


def _expand(records: list[Record], max_windows: int, subset_name: str):
    xs: list[np.ndarray] = []
    ys: list[int] = []
    owners: list[str] = []
    kept: list[Record] = []
    n_bad = 0
    print(f"Extracting {subset_name}: {len(records)} recordings ...", flush=True)
    for i, record in enumerate(records, start=1):
        try:
            features = wav_to_spectrograms(record.path, max_windows=max_windows)
        except Exception as exc:  # one damaged file should not stop the whole run
            n_bad += 1
            print(f"WARNING: skipping {record.unique_id}: {exc}", file=sys.stderr)
            continue
        xs.extend(features)
        ys.extend([record.label] * len(features))
        owners.extend([record.unique_id] * len(features))
        kept.append(record)
        if i % 100 == 0 or i == len(records):
            print(f"  {i}/{len(records)} recordings processed", flush=True)
    if not xs:
        raise RuntimeError(f"No usable examples in {subset_name} split")
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    owner_ids = np.asarray(owners, dtype=str)
    print(f"  created {len(y)} audio windows; skipped {n_bad} recordings", flush=True)
    return X, y, owner_ids, kept, n_bad


def _record_level_predictions(model, X, y, owners, records: list[Record], split_name: str):
    """Average window scores into one understandable row per source recording."""
    window_scores = np.asarray(model.predict(X, batch_size=32, verbose=0)).reshape(-1)
    window_frame = pd.DataFrame({"record_key": owners, "reference_code": y, "score": window_scores})
    grouped = window_frame.groupby("record_key", sort=False).agg(
        reference_code=("reference_code", "first"),
        abnormal_reference_score=("score", "mean"),
        audio_windows_averaged=("score", "size"),
    )
    grouped["predicted_code"] = (grouped["abnormal_reference_score"] >= 0.5).astype(int)
    lookup = {r.unique_id: r for r in records}
    rows = []
    for record_key, row in grouped.iterrows():
        record = lookup[record_key]
        reference_code = int(row["reference_code"])
        predicted_code = int(row["predicted_code"])
        rows.append({
            "recording_filename": record.path.name,
            "dataset_source_group": record.source,
            "data_split": split_name,
            "dataset_reference_label": "Abnormal reference" if reference_code == 1 else "Normal reference",
            "model_classification": "Abnormal-like" if predicted_code == 1 else "Normal-like",
            "model_score_for_abnormal_reference_label": round(float(row["abnormal_reference_score"]), 6),
            "classification_correct": "Yes" if reference_code == predicted_code else "No",
            "number_of_8_second_windows_averaged": int(row["audio_windows_averaged"]),
        })
    return pd.DataFrame(rows)


def _metric_summary(frame: pd.DataFrame) -> dict:
    y_true = frame["_reference_code"].astype(int).to_numpy()
    y_pred = frame["_predicted_code"].astype(int).to_numpy()
    scores = frame["_score"].astype(float).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = None
    return {
        "recordings_evaluated": int(len(frame)),
        "correct_classifications": int(tp + tn),
        "incorrect_classifications": int(fp + fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_abnormal_reference": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_normal_reference": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _internal_metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "_reference_code": (predictions["dataset_reference_label"] == "Abnormal reference").astype(int),
        "_predicted_code": (predictions["model_classification"] == "Abnormal-like").astype(int),
        "_score": predictions["model_score_for_abnormal_reference_label"].astype(float),
    })


def _metric_rows(split_name: str, summary: dict) -> list[dict]:
    definitions = [
        ("recordings_evaluated", "Recordings evaluated", "count", "Number of complete source recordings scored."),
        ("correct_classifications", "Correct classifications", "count", "Recordings where the model's broad class matched the dataset reference label."),
        ("incorrect_classifications", "Incorrect classifications", "count", "Recordings where the model's broad class differed from the dataset reference label."),
        ("accuracy", "Accuracy", "percent", "Percent of recordings classified in agreement with the dataset reference labels."),
        ("balanced_accuracy", "Balanced accuracy", "percent", "Average of sensitivity and specificity; useful when classes are uneven."),
        ("sensitivity_abnormal_reference", "Sensitivity for abnormal-reference class", "percent", "Of recordings labeled abnormal in the dataset, the percent classified abnormal-like."),
        ("specificity_normal_reference", "Specificity for normal-reference class", "percent", "Of recordings labeled normal in the dataset, the percent classified normal-like."),
        ("roc_auc", "ROC AUC", "0_to_1", "How well scores rank abnormal-reference recordings above normal-reference recordings; 0.5 is chance-level ranking."),
    ]
    result = []
    for key, display, unit, explanation in definitions:
        value = summary.get(key)
        if value is not None and unit == "percent":
            value = round(float(value) * 100, 2)
        elif value is not None and unit == "0_to_1":
            value = round(float(value), 4)
        result.append({
            "dataset_split": split_name,
            "measurement": display,
            "value": "Not available" if value is None else value,
            "unit": unit,
            "what_this_means": explanation,
        })
    return result


def _save_human_readable_outputs(run_dir: Path, history, val_predictions: pd.DataFrame,
                                 test_predictions: pd.DataFrame, val_metrics: dict,
                                 test_metrics: dict, record_splits: dict, config: dict,
                                 best_epoch: int, elapsed_seconds: float) -> None:
    val_predictions.to_csv(run_dir / "validation_recordings.csv", index=False)
    test_predictions.to_csv(run_dir / "test_recordings.csv", index=False)
    combined = pd.concat([val_predictions, test_predictions], ignore_index=True)
    combined.to_csv(run_dir / "all_evaluated_recordings.csv", index=False)

    metric_rows = _metric_rows("Validation (use for model selection)", val_metrics)
    metric_rows += _metric_rows("Held-out test (final descriptive check)", test_metrics)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics_summary.csv", index=False)

    test_report = classification_report(
        _internal_metric_frame(test_predictions)["_reference_code"],
        _internal_metric_frame(test_predictions)["_predicted_code"],
        labels=[0, 1], target_names=["Normal reference", "Abnormal reference"],
        output_dict=True, zero_division=0,
    )
    pd.DataFrame([
        {"class": label, "precision_percent": round(values["precision"] * 100, 2),
         "recall_percent": round(values["recall"] * 100, 2),
         "f1_score_percent": round(values["f1-score"] * 100, 2),
         "number_of_recordings": int(values["support"])}
        for label, values in test_report.items() if label in ("Normal reference", "Abnormal reference")
    ]).to_csv(run_dir / "test_class_performance.csv", index=False)

    tn, fp = test_metrics["confusion_matrix"][0]
    fn, tp = test_metrics["confusion_matrix"][1]
    pd.DataFrame([
        {"dataset_reference_label": "Normal reference", "predicted_normal_like": tn, "predicted_abnormal_like": fp},
        {"dataset_reference_label": "Abnormal reference", "predicted_normal_like": fn, "predicted_abnormal_like": tp},
    ]).to_csv(run_dir / "test_confusion_matrix.csv", index=False)

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch_number", np.arange(1, len(history_frame) + 1))
    rename = {
        "loss": "training_loss_binary_crossentropy",
        "val_loss": "validation_loss_binary_crossentropy",
        "accuracy": "training_accuracy_fraction",
        "val_accuracy": "validation_accuracy_fraction",
        "auc": "training_auc",
        "val_auc": "validation_auc",
        "precision": "training_precision_fraction",
        "val_precision": "validation_precision_fraction",
        "recall": "training_recall_fraction",
        "val_recall": "validation_recall_fraction",
    }
    history_frame = history_frame.rename(columns=rename)
    for name in list(history_frame.columns):
        if name.endswith("_accuracy_fraction") or name.endswith("_precision_fraction") or name.endswith("_recall_fraction"):
            history_frame[name.replace("_fraction", "_percent")] = (history_frame[name] * 100).round(2)
            history_frame = history_frame.drop(columns=[name])
    history_frame.to_csv(run_dir / "training_history.csv", index=False)

    config_payload = {
        **config,
        "best_epoch_selected_by_validation_loss": int(best_epoch),
        "epochs_completed": int(len(history.epoch)),
        "training_duration_minutes": round(elapsed_seconds / 60, 2),
        "recording_counts_after_audio_processing": record_splits,
        "validation_metrics": val_metrics,
        "held_out_test_metrics": test_metrics,
        "model_file": "heart_sound_cnn.keras",
    }
    (run_dir / "experiment_details.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def pct(value):
        return "Not available" if value is None else f"{value * 100:.2f}%"

    report_lines = [
        "HEART-SOUND AI — EXPERIMENT REPORT",
        "=" * 45,
        f"Experiment: {config['experiment_id']}",
        f"Started: {config['started_at_local']}",
        f"Training duration: {elapsed_seconds / 60:.1f} minutes",
        "",
        "SETTINGS",
        f"Maximum epochs requested: {config['maximum_epochs_requested']}",
        f"Epochs actually completed: {len(history.epoch)}",
        f"Best epoch (lowest validation loss): {best_epoch}",
        f"Random seed: {config['random_seed']}",
        f"Maximum 8-second windows per recording: {config['maximum_windows_per_recording']}",
        "Split: 70% training / 15% validation / 15% held-out test, by recording",
        "",
        "RECORDINGS USED AFTER AUDIO PROCESSING",
        f"Training: {record_splits['training']}",
        f"Validation: {record_splits['validation']}",
        f"Held-out test: {record_splits['test']}",
        "",
        "VALIDATION RESULTS (use these when choosing settings)",
        f"Accuracy: {pct(val_metrics['accuracy'])}",
        f"Balanced accuracy: {pct(val_metrics['balanced_accuracy'])}",
        f"Sensitivity for abnormal-reference class: {pct(val_metrics['sensitivity_abnormal_reference'])}",
        f"Specificity for normal-reference class: {pct(val_metrics['specificity_normal_reference'])}",
        f"ROC AUC: {val_metrics['roc_auc'] if val_metrics['roc_auc'] is not None else 'Not available'}",
        "",
        "HELD-OUT TEST RESULTS (report once for final evaluation)",
        f"Recordings evaluated: {test_metrics['recordings_evaluated']}",
        f"Accuracy: {pct(test_metrics['accuracy'])}",
        f"Balanced accuracy: {pct(test_metrics['balanced_accuracy'])}",
        f"Sensitivity for abnormal-reference class: {pct(test_metrics['sensitivity_abnormal_reference'])}",
        f"Specificity for normal-reference class: {pct(test_metrics['specificity_normal_reference'])}",
        f"ROC AUC: {test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'Not available'}",
        f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "",
        "FILES TO OPEN",
        "metrics_summary.csv — key measurements with plain-language explanations",
        "training_history.csv — loss and performance after each epoch",
        "test_confusion_matrix.csv — counts of correct/incorrect class matches",
        "test_class_performance.csv — precision, recall, and F1 by reference class",
        "test_recordings.csv — one readable row for each held-out test recording",
        "all_evaluated_recordings.csv — validation and test predictions together",
        "training_history.png — training and validation curves",
        "test_confusion_matrix.png — visual summary of held-out test classifications",
        "heart_sound_cnn.keras — trained model file",
        "",
        "INTERPRETATION AND LIMITATIONS",
        "These are comparisons with broad PhysioNet dataset reference labels, not clinical diagnostic accuracy.",
        "The abnormal-like/normal-like wording describes model output only; it is not a diagnosis.",
        "The random split is by recording, not guaranteed by person, because patient IDs are not consistently available.",
        "Good results on this public dataset do not prove performance with the homemade microphone or on new people.",
        "Use validation results for model selection. Avoid repeatedly changing settings based on held-out test results.",
        "",
        "Educational research only — not for medical decisions.",
    ]
    (run_dir / "EXPERIMENT_REPORT.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    (run_dir / "CSV_FIELD_GUIDE.txt").write_text(
        "CSV FIELD GUIDE\n===============\n\n"
        "metrics_summary.csv\n"
        "  dataset_split: validation data or the held-out test data.\n"
        "  measurement: plain-language name of a performance measurement.\n"
        "  value: measured result. Values with unit=percent are already 0-100, not 0-1.\n"
        "  unit: count, percent, or 0_to_1.\n"
        "  what_this_means: short explanation of the measurement.\n\n"
        "training_history.csv\n"
        "  epoch_number: pass through the training data; starts at 1.\n"
        "  training_loss_binary_crossentropy / validation_loss_binary_crossentropy: model error measure; lower is generally better.\n"
        "  *_accuracy_percent: percent correct on that epoch's windows.\n"
        "  *_auc: ranking measure from 0 to 1; higher is generally better.\n"
        "  *_precision_percent and *_recall_percent: class-performance measures for the model's positive class.\n\n"
        "test_recordings.csv and validation_recordings.csv\n"
        "  recording_filename: audio file name.\n"
        "  dataset_source_group: source subset name within the public dataset.\n"
        "  data_split: validation or held-out test.\n"
        "  dataset_reference_label: label supplied by the research dataset.\n"
        "  model_classification: the model's broad normal-like / abnormal-like output.\n"
        "  model_score_for_abnormal_reference_label: model score from 0 to 1; not a medical probability.\n"
        "  classification_correct: whether the model output agrees with the dataset reference label.\n"
        "  number_of_8_second_windows_averaged: number of audio windows combined for this recording.\n\n"
        "Open CSVs in Excel. If all data appears in one column, use Excel's Data > From Text/CSV import.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an educational heart-sound CNN; each run is preserved.")
    parser.add_argument("--data", type=Path, default=Path("data/physionet2016"), help="Extracted PhysioNet folder")
    parser.add_argument("--out", type=Path, default=Path("experiments"), help="Parent folder for timestamped experiment folders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=3, help="Maximum 8-second windows per recording")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None, help="Optional short label for this experiment")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error("--epochs must be from 1 to 100")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    run_dir, experiment_id = _make_experiment_dir(args.out, args.run_name, args.seed, args.epochs)
    started = time.time()
    started_local = datetime.now().astimezone().isoformat(timespec="seconds")
    config = {
        "experiment_id": experiment_id,
        "started_at_local": started_local,
        "maximum_epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "random_seed": args.seed,
        "maximum_windows_per_recording": args.max_windows,
        "target_sample_rate_hz": 2000,
        "segment_duration_seconds": 8,
        "split_percentages": {"training": 70, "validation": 15, "held_out_test": 15},
        "dataset_folder": str(args.data),
        "tensorflow_version": tf.__version__,
    }
    (run_dir / "experiment_details.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nEXPERIMENT: {experiment_id}")
    print(f"All results will be saved in: {run_dir.resolve()}\n")

    records = load_records(args.data)
    labels = np.asarray([r.label for r in records], dtype=int)
    counts = {"normal_reference": int(np.sum(labels == 0)), "abnormal_reference": int(np.sum(labels == 1))}
    print(f"Found {len(records)} labeled recordings: {counts}")
    if min(counts.values()) < 6:
        raise RuntimeError("Need at least six recordings of each reference class to create stratified splits.")

    # Split whole recordings first to prevent windows from one file crossing subsets.
    train_records, temporary_records = train_test_split(
        records, test_size=0.30, random_state=args.seed, stratify=labels
    )
    temporary_labels = np.asarray([r.label for r in temporary_records], dtype=int)
    val_records, test_records = train_test_split(
        temporary_records, test_size=0.50, random_state=args.seed, stratify=temporary_labels
    )

    X_train, y_train, ids_train, records_train, bad_train = _expand(train_records, args.max_windows, "training")
    X_val, y_val, ids_val, records_val, bad_val = _expand(val_records, args.max_windows, "validation")
    X_test, y_test, ids_test, records_test, bad_test = _expand(test_records, args.max_windows, "held-out test")

    if tuple(X_train.shape[1:]) != INPUT_SHAPE:
        raise RuntimeError(f"Model input mismatch: got {X_train.shape[1:]}, expected {INPUT_SHAPE}")
    for subset_name, subset_y in (("training", y_train), ("validation", y_val), ("test", y_test)):
        if len(np.unique(subset_y)) < 2:
            raise RuntimeError(f"Both reference classes must remain represented in the {subset_name} subset after unreadable files are skipped.")

    inventory_rows = []
    source_counts: dict[str, dict[str, int]] = {}
    for r in records:
        label_name = "Abnormal reference" if r.label == 1 else "Normal reference"
        source_counts.setdefault(r.source, {"Normal reference": 0, "Abnormal reference": 0})[label_name] += 1
    for source, class_counts in sorted(source_counts.items()):
        for label_name, count in class_counts.items():
            inventory_rows.append({"dataset_source_group": source, "dataset_reference_label": label_name, "recording_count": count})
    pd.DataFrame(inventory_rows).to_csv(run_dir / "dataset_inventory.csv", index=False)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    model = build_model(INPUT_SHAPE)
    print("\nNeural-network structure:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    print("\nTraining started. Early stopping will retain the best validation-loss weights.")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    best_epoch = int(np.argmin(history.history.get("val_loss", [0])) + 1)
    final_model_path = run_dir / "heart_sound_cnn.keras"
    model.save(final_model_path)

    val_predictions = _record_level_predictions(model, X_val, y_val, ids_val, records_val, "validation")
    test_predictions = _record_level_predictions(model, X_test, y_test, ids_test, records_test, "held-out test")
    val_internal = _internal_metric_frame(val_predictions)
    test_internal = _internal_metric_frame(test_predictions)
    val_metrics = _metric_summary(val_internal)
    test_metrics = _metric_summary(test_internal)
    record_splits = {
        "training": len(records_train),
        "validation": len(records_val),
        "test": len(records_test),
        "skipped_unreadable_training": bad_train,
        "skipped_unreadable_validation": bad_val,
        "skipped_unreadable_test": bad_test,
    }
    elapsed = time.time() - started
    _save_human_readable_outputs(
        run_dir, history, val_predictions, test_predictions, val_metrics, test_metrics,
        record_splits, config, best_epoch, elapsed,
    )

    # Accuracy/loss curves are based on windows, while the CSV metrics are recording-level.
    hist = history.history
    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("loss", []), label="Training loss")
    plt.plot(hist.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss (lower is better)")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_history.png", dpi=160)
    plt.close()

    cm = np.asarray(test_metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Held-out test confusion matrix")
    plt.colorbar(label="Number of recordings")
    plt.xticks([0, 1], ["Predicted normal-like", "Predicted abnormal-like"], rotation=15, ha="right")
    plt.yticks([0, 1], ["Dataset normal reference", "Dataset abnormal reference"])
    for (r, c), value in np.ndenumerate(cm):
        plt.text(c, r, str(value), ha="center", va="center")
    plt.ylabel("Dataset reference label")
    plt.xlabel("Model output at a 0.5 threshold")
    plt.tight_layout()
    plt.savefig(run_dir / "test_confusion_matrix.png", dpi=160)
    plt.close()

    metadata = {
        "model_filename": final_model_path.name,
        "input_shape": list(INPUT_SHAPE),
        "target_sample_rate_hz": 2000,
        "segment_seconds": 8,
        "max_windows_per_recording": args.max_windows,
        "threshold": 0.5,
        "class_0": "dataset reference-normal label",
        "class_1": "dataset reference-abnormal label",
        "educational_use_only": True,
    }
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # The latest pointer makes menu option 4 convenient; prior run folders are untouched.
    latest_pointer = args.out.resolve() / "latest_run.txt"
    latest_pointer.write_text(experiment_id + "\n", encoding="utf-8")
    print("\nTRAINING FINISHED")
    print(f"Best epoch selected by validation loss: {best_epoch}")
    print(f"Held-out test accuracy (recording-level): {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Results folder: {run_dir.resolve()}")
    print("Start with EXPERIMENT_REPORT.txt and metrics_summary.csv.")
    print("Use validation results when selecting settings; do not tune repeatedly against test results.")
    print("Educational research only — not clinical diagnostic accuracy.")


if __name__ == "__main__":
    main()
