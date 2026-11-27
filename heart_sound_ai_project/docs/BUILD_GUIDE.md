# Physical Build & Setup Guide

## 1. What to build

The simplest version uses a stethoscope as the acoustic path, an INMP441 digital microphone as the sensor, an ESP32 as the recorder/network link, and your Windows computer for the AI. Your Raspberry Pi 4 is optional; it can run the receiver, but the Windows computer is the easier place to install TensorFlow and train the model.

**Recommended path:** build the USB-powered ESP32 microphone first. Do not add a battery or permanent enclosure until the recording works.

## 2. Exact parts list

### Required to assemble the recorder

| Qty. | Part | What to look for | Notes |
|---:|---|---|---|
| 1 | Classic ESP32 development board | ESP32-WROOM-32 / ESP32 DevKit-style board with exposed GPIO pins and USB connection | Use a conventional ESP32 supported by the supplied Arduino firmware. Avoid substituting a Pico W or ESP32-C3/S3 without changing and testing the firmware. |
| 1 | INMP441 microphone breakout | INMP441 **I²S digital MEMS microphone module**, with pads/pins labeled VDD, GND, SCK, WS, SD, L/R | Prefer the pre-soldered-header version. Do not buy an analog MAX9814/MAX4466 module for this design. |
| 1 | Acoustic stethoscope | Ordinary stethoscope with chestpiece and flexible tubing | Reuse one you already have if available. |
| 1 | Solderless breadboard | Half-size or full-size | For temporary wiring. |
| 1 pack | Jumper wires | Male-to-female and male-to-male as needed by your ESP32 header and microphone board | Short wires are preferable for I²S connections. |
| 1 | USB data cable | Connector that fits your ESP32 board (often USB-C or Micro-USB) | Must support data transfer, not charging only. |
| 1 | Computer | Windows/macOS/Linux with Python 3.11 | Runs the Python receiver and TensorFlow. Your existing laptop/desktop is fine. |

### Recommended inexpensive acoustic-mechanical supplies

| Qty. | Part | Purpose |
|---:|---|---|
| Small piece | Soft silicone tubing | Removable connection from one stethoscope earpiece outlet to the mic coupler. Measure your actual tubing before buying. |
| 1–2 | Silicone O-rings / soft gasket | Helps seal the coupler without covering the microphone port. |
| Small piece | Closed-cell foam or soft gasket foam | Supports the board and reduces rattling. Keep it away from the mic port. |
| As needed | Cardboard/plastic prototype pieces or 3D-printed PLA/TPU holder | Holds the coupler after the first test works. |
| A few | Small zip ties / removable tape | Strain relief and temporary mounting. Never tape over the mic port. |

### Do not buy for version 1

- LiPo battery or TP4056 charger (use USB power first).
- Analog-to-digital converter (INMP441 already produces digital I²S data).
- Analog switch.
- External AI accelerator.
- A Raspberry Pi (you already have one; also optional if using the computer).
- A 3D printer or custom case before the acoustic path is tested.

## 3. Purchase and reference links

- [Espressif ESP32 DevKitC documentation](https://documentation.espressif.com/projects/arduino-esp32/en/latest/boards/ESP32-DevKitC-1.html) — check the board family and pin labels.
- [INMP441 manufacturer information (TDK/InvenSense)](https://product.tdk.com/en/search/sw_piezo/mic/mems-mic/info?part_no=INMP441) — confirm the exact microphone model.
- [Example pre-soldered INMP441 breakout listing](https://shillehtek.com/products/inmp441-i2s-omnidirectional-microphone-module-with-soldering) — product example, not an endorsement; verify that the selected listing is the I²S INMP441 breakout and that header pins are included/soldered.
- [Arduino-ESP32 I²S documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/i2s.html) — reference for the digital audio interface.
- [Official PhysioNet dataset](https://physionet.org/content/challenge-2016/1.0.0/) — dataset used for model training, not a hardware component.

Prices and stock change. The minimum shopping list is one conventional ESP32 board, one INMP441 breakout, jumper wires, and a USB data cable if you do not already own them. Reuse your stethoscope, computer, breadboard, and other supplies where possible.

## 4. Wire the INMP441 to the ESP32

Disconnect the ESP32 from USB before changing wires. Connect according to the labels on your exact breakout:

| INMP441 module | Classic ESP32 DevKit | Purpose |
|---|---|---|
| VDD / 3V | 3V3 | 3.3 V power only |
| GND | GND | Shared ground |
| SCK / BCLK | GPIO 26 | I²S bit clock |
| WS / LRCL | GPIO 25 | I²S word-select clock |
| SD / DOUT | GPIO 33 | Microphone digital data into ESP32 |
| L/R / SEL | GND | Selects the left channel/slot for this firmware |

**Do not connect the INMP441 VDD to 5 V.** Check the module's printed labels; the order of pins varies between breakout boards. GPIO 26, 25, and 33 are the defaults in the provided sketch. Do not use ESP32 flash-connected pins GPIO 6–11.

Open `hardware/wiring_diagram.png` or `hardware/wiring_diagram.svg` for the visual guide. Have a teacher/adult review the wiring before connecting power. Keep water and liquids away from the electronics.

## 5. Physically couple the microphone to the stethoscope

The INMP441 is a tiny bottom-port microphone. The actual acoustic hole may be on the underside of your breakout board, not the side with the pin labels. Inspect the specific board before designing the holder.

A simple first coupler uses one stethoscope earpiece outlet as the sound source:

1. Lay the stethoscope on a table. Do not wear it during this test.
2. Choose one earpiece tube/outlet and make a removable connection using a short piece of soft silicone tubing or a snug adapter. Do not cut or permanently modify the stethoscope.
3. Make a small air cavity at the other end of the adapter. Position the microphone's acoustic port facing into that cavity with a small air gap. The opening must remain unobstructed.
4. Use a soft O-ring/gasket around the outside of the cavity to reduce air leaks. The gasket may touch the holder, but it must not cover the microphone hole, microphone package, or sound entry path.
5. Secure the PCB by its edges or mounting holes. Add strain relief to the wires so tugging the cable cannot pull on the microphone board.
6. Begin with removable tape/foam/cardboard. Do not pour silicone sealant over the microphone or electronics. Do not glue the microphone into a sealed blob.
7. Only after the removable design records correctly, make a dimensioned 3D-printed holder based on measurements of your exact board, tube, and gasket. The SVG coupler image is conceptual, not a dimensioned print file.

If the audio is quiet, first check pinout, 3.3 V power, port orientation, and whether the air path is open. Do not solve a mechanical issue by increasing gain or sealing over the port.

## 6. First test: use the audio files included in this package

The `demo_audio/` folder contains two short synthetic WAV files. They are artificial PCG-like test signals, not recorded hearts and not labeled normal/abnormal.

1. Connect a speaker to your computer and play `synthetic_steady_timing_demo.wav` at a fixed, moderate volume.
2. Hold the stethoscope chestpiece near the speaker (not against a person) and listen through its tubing, or use the tubing/coupler path you are testing.
3. Run menu option 5 to make waveform and spectrogram reports.
4. Repeat with the second synthetic clip at the same speaker distance and volume.
5. Use the same speaker position, volume, and setup when comparing different coupler designs.

These synthetic clips are for recording-chain checks only. Never use them for model training, held-out model testing, or a claim about heart abnormality.

## 7. Install the software on Windows

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/). During installation, choose the option to add Python to PATH.
2. Extract the project ZIP to a normal folder such as `Documents\\HeartSoundAI`.
3. Double-click `START_HERE_WINDOWS.bat`.
4. The script creates `.venv`, installs the libraries listed in `requirements.txt`, and opens the numbered menu. The first package install requires an internet connection.

For macOS/Linux, install Python 3.11 and run `./START_HERE_MAC_LINUX.sh` from Terminal.

**Training note:** use your laptop/desktop for TensorFlow training. TensorFlow installation on Raspberry Pi OS can depend on OS and processor architecture. You can keep the Pi for later experiments; it is not required for this design.

## 8. Download the real research dataset and train

In the menu:

1. Choose **1** to download the official PhysioNet/CinC Challenge 2016 training archive. The compressed file is approximately 181 MB; it expands into a much larger folder. Allow disk space for the extracted audio and model outputs.
2. After the downloader verifies the ZIP and extracts it, choose **2** to train. Start with 5 or 10 epochs to check that everything works; use 20–30 epochs for a more complete experiment if compute time allows.
3. Each run creates a separate, timestamped folder inside `experiments/`. Nothing from an earlier completed run is overwritten. Open that run folder and start with `EXPERIMENT_REPORT.txt` or `metrics_summary.csv`. For more detail, see `training_history.csv`, `test_recordings.csv`, `test_confusion_matrix.csv`, `test_class_performance.csv`, `training_history.png`, and `test_confusion_matrix.png`. `CSV_FIELD_GUIDE.txt` explains the columns in plain language.
4. To compare completed runs, return to the menu and choose option 6. It creates `experiments/experiment_comparison.csv`. Use validation accuracy/balanced accuracy as the main comparison for selecting settings; treat test metrics as final descriptive checks, not values to repeatedly optimize.

The downloader writes `DATASET_ATTRIBUTION.txt` into the data folder. Preserve attribution/license information if using results or distributing derived materials. The full dataset is fetched from PhysioNet on your computer; it is not contained in this project ZIP because it is a separate large third-party dataset.

## 9. Flash the ESP32 recorder firmware

1. Install Arduino IDE 2.x from the official Arduino site.
2. Add the Espressif ESP32 board package using the [official Arduino-ESP32 installation instructions](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html).
3. Open `firmware/esp32_inmp441/esp32_inmp441.ino`.
4. In the sketch, edit `WIFI_SSID` and `WIFI_PASSWORD` for a trusted private 2.4 GHz Wi-Fi network.
5. Start menu option **3** on your computer to launch the receiver. Find the computer's local IPv4 address (for Windows, open Command Prompt and run `ipconfig`; use the IPv4 address of the active Wi-Fi adapter). Replace the example IP in `SERVER_URL` with `http://YOUR_COMPUTER_IP:8000/upload`.
6. Keep the computer and ESP32 on the same private Wi-Fi network. Do not use a public hotspot or a guest network that isolates devices.
7. In Arduino IDE, select a classic ESP32 Dev Module board and the correct COM port. Upload the sketch.
8. Open Serial Monitor at 115200 baud. Type `r` and press Enter. The ESP32 captures one 8-second audio clip and uploads it.
9. The receiver saves the file under `recordings/`. Run menu option **5** to inspect its waveform and spectrogram.

If Windows Firewall asks whether Python may communicate on the network, only allow it on a trusted private network. Do not create router port-forwarding rules or expose the receiver publicly. The receiver can be stopped with Ctrl+C.

## 10. Recommended order of work

1. Install Python and open the menu.
2. Generate the signal reports for the bundled synthetic WAV clips.
3. Download PhysioNet and train the model on the laptop.
4. Wire the ESP32 and microphone on the breadboard.
5. Record controlled speaker playback and verify a saved WAV.
6. Test the removable stethoscope acoustic coupler and compare repeated recordings.
7. Only then design a permanent printed holder and prepare the science-fair display.

This order separates software, electronics, and acoustic problems, so debugging is much easier.

## 11. Troubleshooting

| Problem | Check |
|---|---|
| `py -3.11` not found | Install Python 3.11 and rerun the launcher. |
| TensorFlow cannot install | Confirm Python 3.11 and 64-bit operating system; run training on your Windows computer rather than the Pi. |
| Download fails | Confirm internet access and retry menu option 1; alternatively download `training.zip` manually from the official PhysioNet page and follow its extraction instructions. |
| Dataset not found | Check that `data/physionet2016/` contains nested `REFERENCE.csv` files and WAVs. Run menu option 1 again if it is incomplete. |
| ESP32 cannot connect to Wi-Fi | Check SSID/password and use a trusted 2.4 GHz network. |
| Server does not receive upload | Ensure menu option 3 is running, `SERVER_URL` contains the computer's correct local IP, both devices share the same Wi-Fi, and firewall allows private-network access. |
| WAV is almost silent | Check VDD=3.3 V, GND, L/R=GND, SCK/WS/SD connections, microphone hole orientation, and unobstructed air path. |
| Audio is distorted | Check mechanical rattling, handling movement, clipping, loose wires, and speaker volume. |
| Server saves WAV but has no model score | Train the model first. Confirm that `experiments/latest_run.txt` points to a completed run folder containing `heart_sound_cnn.keras`, then restart the receiver. |

## 12. Safety and scope

This is a school research prototype, not a clinical instrument. The reference labels classify broad dataset recording groups and do not identify a specific disease. A model score is not a calibrated medical probability. Do not use the device on people to make health claims or decisions. Use the synthetic project audio and publicly available research recordings for controlled playback tests. Any study involving human participants or personal health recordings needs prior school/teacher and parent/guardian approval and any required institutional process.
