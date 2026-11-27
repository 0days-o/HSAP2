# Science Fair Research Plan

## Recommended research question

**How accurately can a low-cost digital recording prototype and a convolutional neural network classify public heart-sound recordings against the dataset's reference labels?**

The original question asked whether the system can match trained listeners. That comparison cannot be claimed unless suitably trained listeners independently classify the same held-out recordings under an approved, consistent protocol. The supplied software evaluates the model against the dataset's reference labels instead.

## Hypothesis

A low-cost digital recording chain paired with a CNN trained on public heart-sound recordings will learn patterns associated with the dataset's normal-reference and abnormal-reference labels. Performance may be lower on audio from the homemade microphone because the training files were captured using different equipment and in different environments.

Do not choose an expected accuracy number before the experiment. A result below expectation is still a valid result.

## Independent, dependent, and controlled variables

| Variable | Example | How to manage it |
|---|---|---|
| Independent variable (hardware characterization) | Coupler condition: loose fit versus gasketed fit | Change only the coupler; keep speaker distance and volume constant. |
| Independent variable (software experiment) | Preprocessing/model setting | Change one setting at a time and record it. |
| Dependent variable (hardware) | Recording RMS, peak amplitude, clipping fraction, waveform/spectrogram consistency | Use the reports in `signal_reports/`. |
| Dependent variable (software) | Held-out recording-level accuracy, balanced accuracy, sensitivity, specificity, ROC AUC | Read the readable `metrics_summary.csv` and `EXPERIMENT_REPORT.txt` inside that run's folder under `experiments/`. |
| Controlled variables | Public audio source, sample rate, segment duration, data split seed, model threshold, speaker position/volume | Keep fixed during the main experiment. |

## Audio and dataset

The project ZIP includes two **synthetic PCG-like WAV files** only for immediate audio-pipeline and hardware playback checks. They are mathematically generated examples—not recordings of people, not real heart sounds, and not normal/abnormal labels. Do not train or evaluate the classifier on them.

For model training, use the official [PhysioNet/CinC Challenge 2016 dataset](https://physionet.org/content/challenge-2016/1.0.0/). Menu option 1 downloads and extracts its training archive. The dataset provides broad reference labels (`-1` normal and `+1` abnormal). These are dataset-level labels, not specific diagnoses or clinical gold standards for this prototype. Keep the generated `DATASET_ATTRIBUTION.txt` and follow the dataset's ODC-BY 1.0 license requirements.

## AI experiment

The supplied software:

1. Loads WAV recordings and their challenge reference labels.
2. Resamples recordings to 2 kHz, applies a broad 25–800 Hz band-pass filter, and forms 8-second audio windows.
3. Converts each window to a standardized log-Mel spectrogram.
4. Splits recordings into training, validation, and test groups before extracting windows.
5. Trains a small TensorFlow/Keras CNN.
6. Averages window scores back to a recording-level score and evaluates the held-out test recordings.
7. Saves each run in a separate timestamped folder, including the model, settings, readable CSV outputs, and a plain-language report.
8. Allows experiment-to-experiment comparison, using validation metrics for model selection and reserving the held-out test set for final descriptive evaluation.

Report accuracy and balanced accuracy together because the source dataset is class-imbalanced. Also report sensitivity for the abnormal-reference class, specificity for the normal-reference class, ROC AUC when it can be calculated, and the confusion matrix. Do not report training accuracy as test performance. For choosing between settings, use validation metrics. Repeatedly choosing settings based on held-out test results gradually turns the test set into another validation set and weakens the final evaluation.

The split is performed by recording, not by person. The reference files do not consistently expose subject identifiers that would allow a guaranteed patient-independent split. State this limitation clearly.

## Hardware characterization experiment

This experiment checks audio capture quality, not whether a person has a heart condition.

1. Play one included synthetic audio test file or a public PhysioNet WAV through the same speaker at a fixed volume.
2. Keep the speaker and stethoscope position fixed.
3. Capture at least five trials with one acoustic-coupler condition.
4. Repeat for another coupler condition, changing nothing else.
5. Use menu option 5 to create waveform/spectrogram plots and a CSV of signal measurements.
6. Compare recording level, clipping fraction, waveform consistency, and visible frequency content.

Use the same source audio for every repeat. Do not interpret a model score as a measurement of coupler quality. Keep the tests to controlled speaker playback; do not collect classmates' or family members' body sounds.

## Result statement template

> On a held-out test set of **[N] public recordings**, the model achieved **[accuracy] accuracy** and **[balanced accuracy] balanced accuracy** against the dataset's reference labels. Sensitivity for the abnormal-reference class was **[value]**, and specificity for the normal-reference class was **[value]**. The confusion matrix contained **[FP] false positives** and **[FN] false negatives**. These values measure agreement with the dataset's broad reference labels, not clinical diagnostic accuracy. The homemade microphone was not clinically validated, and the source files do not support a guaranteed patient-independent split.

## Safety, privacy, and interpretation

- The prototype is for school research and education, not medical diagnosis or treatment decisions.
- Do not interpret its output for yourself or another person. A model score is not a calibrated medical probability.
- Use the bundled synthetic signals and public dataset recordings for speaker-playback testing.
- Do not record people or collect health information unless your teacher, parent/guardian, and any required school review process have approved the exact activity.
- Keep recordings local and do not include names or identifying details in filenames.
- The local receiver should remain on a trusted private Wi-Fi network; do not expose it to the public internet.

## Dataset citation

Liu C, Springer D, Li Q, Moody B, Juan RA, Chorro FJ, Castells F, Roig JM, Silva I, Johnson AE, Syed Z, Schmidt SE, Papadaniil CD, Hadjileontiadis L, Naseri H, Moukadem A, Dieterlen A, Brandt C, Tang H, Samieinasab M, Sameni R, Sameni MR, Mark RG, Clifford GD. An open access database for the evaluation of heart sound algorithms. *Physiological Measurement*. 2016;37(12):2181–2213. https://doi.org/10.1088/0967-3334/37/12/2181
