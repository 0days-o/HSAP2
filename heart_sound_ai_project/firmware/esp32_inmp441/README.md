# ESP32 + INMP441 Firmware

This sketch records one 8-second clip on serial command `r` and uploads signed 16-bit, 8 kHz mono PCM to the Python receiver at `/upload`.

## Before flashing

1. Use a classic ESP32 DevKit/WROOM-style board (the pin mapping may need changes for ESP32-S3 or other variants).
2. Wire the INMP441 according to `../../hardware/wiring_diagram.svg`.
3. Install a current Arduino-ESP32 3.x board package so `ESP_I2S.h` is available.
4. Set Wi-Fi name/password, the receiver host's private LAN IP, and the matching optional token.
5. Compile and flash. Open Serial Monitor at 115200 baud and type `r`.

## Data alignment note

The INMP441 provides 24-bit audio within a 32-bit I²S slot. The code keeps the upper 16 bits (`raw >> 16`) for PCM16 transmission. Verify the saved waveform amplitude on your exact breakout. If the signal is nearly silent or consistently clipped, stop and inspect I²S configuration and wiring before using the data.

## Do not use public Wi-Fi / internet-facing receiver

This is an unauthenticated-by-default classroom prototype unless you set a token. Use a private LAN and do not port-forward port 8000.
