# Heart-Sound AI Science Fair Project

**Single-package starter kit · Version 1.1**  
**Purpose:** educational audio recording, signal-processing, and machine-learning experiment.  
**Important:** this is not a medical device. Never use its audio, scores, or classifications to diagnose, rule out, or treat a health condition.

## Start here — no coding required for the first setup

### Windows
1. Install **Python 3.11** from [python.org](https://www.python.org/downloads/). During setup, enable the option to add Python to PATH.
2. Extract this ZIP into a normal folder (for example, `Documents\HeartSoundAI`). Do not run it from inside the ZIP preview.
3. Double-click `START_HERE_WINDOWS.bat`.
4. The first launch creates a private Python environment and installs the listed packages. It then opens a numbered menu.
5. Choose **1** once to download the public PhysioNet training dataset. After it finishes, choose **2** to train the model.

### macOS / Linux
Install Python 3.11, open Terminal in this folder, and run:

```bash
chmod +x START_HERE_MAC_LINUX.sh
./START_HERE_MAC_LINUX.sh
```

The first package installation requires internet access. The full PhysioNet data download is a separate one-time step in the menu (about 181 MB compressed). The project ZIP includes two synthetic WAV test signals so you can test audio-file loading and plotting immediately, before downloading the research dataset.

## What is included

- `main.py` — numbered menu for dataset download, training, receiver, WAV analysis, and audio reports.
- `pcg_ai/` — audio preprocessing, label loading, CNN model, experiment-tracked training/evaluation, experiment comparison, prediction, receiver, dataset downloader, and synthetic demo audio generator.
- `demo_audio/` — two **synthetic** PCG-like WAVs for audio-pipeline testing only. They are not real recordings and are not suitable for training or accuracy evaluation.
- `firmware/esp32_inmp441/` — ESP32 firmware for one-shot recording and Wi-Fi upload.
- `hardware/` — wiring and acoustic coupler concept drawings (the coupler drawing is not a dimensioned CAD file).
- `docs/BUILD_GUIDE.md` — physical build instructions, full parts list, software setup, and troubleshooting.
- `docs/SCIENCE_FAIR_METHODS.md` — research question, hypothesis, variables, evaluation plan, and limitations.
- `tests/` — signal-preprocessing and experiment-reporting tests.

## The architecture

**Stethoscope tube → removable acoustic coupler → INMP441 I²S microphone → ESP32 → local Wi-Fi → Python receiver on your Windows computer / Mac / Linux computer → TensorFlow model.**

Your Raspberry Pi 4 can also host the receiver, but for the least complicated setup, run the Python software on your Windows laptop/desktop and connect the ESP32 to the same private Wi-Fi network. The Pi is optional. A Pico/Pico W is not used by the supplied firmware; use a conventional ESP32 development board for this version.

## Menu options

1. Download the official PhysioNet/CinC Challenge 2016 training data. The script downloads and unpacks the archive into `data/physionet2016/` and saves attribution details.
2. Train the CNN and create validation and held-out test reports. Every run receives a new timestamped folder under `experiments/`; earlier runs are not overwritten.
3. Start the LAN receiver. It waits for an ESP32 recording upload and saves a WAV under `recordings/`. It automatically loads the latest completed experiment model.
4. Run the latest trained model on a selected WAV file.
5. Create waveform, spectrogram, and recording-chain summaries for the synthetic demo files and any saved recordings.
6. Compare all saved training experiments and create `experiments/experiment_comparison.csv`.
7. Exit.

Training requires the full public dataset and takes longer than audio plotting. Start with 5–10 epochs for a quick software check; use more epochs only after the pipeline works. Each training run is saved separately, including its model, settings, CSV files, plots, and plain-language report. You can give runs different random seeds or labels from the menu.

### Understand your experiment results

After option 2 finishes, open the new folder inside `experiments/`. Start with `EXPERIMENT_REPORT.txt` for a plain-language summary and `metrics_summary.csv` for the key measurements plus explanations. `training_history.csv` shows progress epoch by epoch; `test_recordings.csv` gives one readable row per held-out recording; `test_confusion_matrix.csv` shows the correct/incorrect counts. `CSV_FIELD_GUIDE.txt` explains the columns.

Choose option 6 to compare training runs. Use validation metrics as the main guide when selecting model settings. The test results are included for transparent final reporting, but should not be used repeatedly to tune settings. The latest completed model is automatically selected by options 3 and 4.

## Dataset and research interpretation

The official PhysioNet/CinC Challenge 2016 training set contains 3,126 recordings and broad normal/abnormal reference labels. The labels are not specific diagnoses. The model evaluates against those dataset reference labels; it does **not** establish what a sound means for a person. Read the official [dataset page and license](https://physionet.org/content/challenge-2016/1.0.0/) and retain the generated `DATASET_ATTRIBUTION.txt` if you share derived results.

Citation: Liu C, Springer D, Li Q, Moody B, Juan RA, Chorro FJ, Castells F, Roig JM, Silva I, Johnson AE, Syed Z, Schmidt SE, Papadaniil CD, Hadjileontiadis L, Naseri H, Moukadem A, Dieterlen A, Brandt C, Tang H, Samieinasab M, Sameni R, Sameni MR, Mark RG, Clifford GD. “An open access database for the evaluation of heart sound algorithms.” *Physiological Measurement*. 2016;37(12):2181–2213. https://doi.org/10.1088/0967-3334/37/12/2181

## Important limitations

- The supplied WAV demos are generated mathematically. They are not human/clinical data, are not labeled, and must never be used to train or evaluate the model.
- The full research dataset is downloaded from its official source by menu option 1. It is not embedded in this ZIP.
- Test metrics compare the model with the challenge dataset's broad reference labels. They are not clinical sensitivity/specificity and are not a comparison with trained listeners.
- A recording-level split is used before windows are extracted, but the public reference files do not consistently expose subject IDs; patient-independent splitting cannot be guaranteed.
- The model is trained on source recordings from different equipment and environments. Good dataset metrics do not establish that the homemade microphone captures equivalent audio or that the device works on people.
- Use controlled playback of public recordings through a speaker to characterize the microphone/coupler. Do not collect recordings from classmates or family members without appropriate teacher, parent/guardian, and school approval.
- Keep the Flask receiver on a trusted private local network. Do not port-forward it or expose it to the public internet.

For assembly, start with `docs/BUILD_GUIDE.md`.
