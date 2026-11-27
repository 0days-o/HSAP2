/*
  Educational digital heart-sound recorder
  Board: classic ESP32 DevKit / ESP32-WROOM-style board
  Microphone: INMP441 I2S MEMS module

  The ESP32 captures 8 seconds of 8 kHz, 16-bit mono PCM and POSTs it to the
  Python receiver on a private LAN. Trigger one capture by typing 'r' in Serial
  Monitor. It does not record continuously.

  SAFETY: research/education only. Do not use the output for diagnosis or care.
*/
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ESP_I2S.h>
#include <vector>

// ---- EDIT THESE SETTINGS BEFORE UPLOADING ----
const char *WIFI_SSID = "REPLACE_WITH_WIFI_NAME";
const char *WIFI_PASSWORD = "REPLACE_WITH_WIFI_PASSWORD";
// Replace with the LAN IP address of the computer or Raspberry Pi running server.py.
// Example: "http://192.168.1.80:8000/upload"
const char *SERVER_URL = "http://192.168.1.80:8000/upload";
// Optional. Must match PCG_UPLOAD_TOKEN on the server. Leave empty on both sides to disable.
const char *UPLOAD_TOKEN = "";

// I2S pins. These are examples for a classic ESP32 DevKit; avoid flash pins GPIO 6-11.
constexpr int I2S_BCLK = 26; // microphone SCK / bit clock
constexpr int I2S_WS   = 25; // microphone WS / LRCLK
constexpr int I2S_SD   = 33; // microphone SD / data out -> ESP32 DIN

constexpr uint32_t SAMPLE_RATE = 8000;
constexpr uint32_t RECORD_SECONDS = 8;
constexpr size_t SAMPLE_COUNT = SAMPLE_RATE * RECORD_SECONDS;

I2SClass I2S;

bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  Serial.printf("Connecting to Wi-Fi: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi connected. ESP32 IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }
  Serial.println("Wi-Fi connection failed. Check the SSID/password and 2.4 GHz network.");
  return false;
}

bool setupMicrophone() {
  // INMP441 L/R must be connected to GND to select the left I2S slot.
  // The module sends 24 useful bits in a 32-bit I2S slot; keeping the upper
  // 16 bits gives signed 16-bit PCM adequate for this classroom demo.
  I2S.setPins(I2S_BCLK, I2S_WS, -1, I2S_SD);
  if (!I2S.begin(I2S_MODE_STD, SAMPLE_RATE,
                 I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("ERROR: I2S start failed. Verify the Arduino-ESP32 core and wiring.");
    return false;
  }
  return true;
}

bool recordAndUpload() {
  if (!connectWiFi()) return false;

  Serial.printf("Recording %lu seconds at %lu Hz... keep the sensor still.\n",
                (unsigned long)RECORD_SECONDS, (unsigned long)SAMPLE_RATE);
  std::vector<int16_t> pcm;
  try {
    pcm.resize(SAMPLE_COUNT);
  } catch (...) {
    Serial.println("ERROR: Not enough RAM for the recording buffer.");
    return false;
  }

  uint64_t sumSquares = 0;
  size_t filled = 0;
  while (filled < SAMPLE_COUNT) {
    int32_t raw = 0;
    size_t bytesRead = I2S.readBytes(reinterpret_cast<char *>(&raw), sizeof(raw));
    if (bytesRead != sizeof(raw)) continue;
    const int16_t sample = static_cast<int16_t>(raw >> 16);
    pcm[filled++] = sample;
    sumSquares += (int32_t)sample * (int32_t)sample;
  }
  const double rms = sqrt((double)sumSquares / (double)SAMPLE_COUNT);
  Serial.printf("Capture complete. PCM RMS: %.1f (0-32768 scale)\n", rms);
  if (rms < 3.0) {
    Serial.println("WARNING: The signal is almost silent. Check mic power, I2S pins, and mic port orientation.");
  }

  HTTPClient http;
  http.setTimeout(15000);
  if (!http.begin(SERVER_URL)) {
    Serial.println("ERROR: Could not open the receiver URL.");
    return false;
  }
  http.addHeader("Content-Type", "application/octet-stream");
  if (UPLOAD_TOKEN[0] != '\0') http.addHeader("X-Upload-Token", UPLOAD_TOKEN);

  const uint8_t *payload = reinterpret_cast<const uint8_t *>(pcm.data());
  const size_t payloadBytes = pcm.size() * sizeof(int16_t);
  Serial.printf("Uploading %u bytes to %s\n", (unsigned)payloadBytes, SERVER_URL);
  const int httpCode = http.POST(payload, payloadBytes);
  const String response = http.getString();
  http.end();

  Serial.printf("Receiver HTTP status: %d\n", httpCode);
  if (response.length()) Serial.println(response);
  if (httpCode >= 200 && httpCode < 300) {
    Serial.println("Upload complete. Review the saved WAV and research-demo score on the receiver.");
    return true;
  }
  Serial.println("Upload failed. Check server.py, host IP, firewall, network, and token.");
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println();
  Serial.println("Heart-sound recorder — educational use only, not a medical device.");
  if (!setupMicrophone()) {
    Serial.println("Restart after fixing the I2S configuration.");
    while (true) delay(1000);
  }
  connectWiFi();
  Serial.println("Ready. Type r then press Enter to record and upload one 8-second clip.");
}

void loop() {
  while (Serial.available()) {
    const char c = (char)Serial.read();
    if (c == 'r' || c == 'R') recordAndUpload();
  }
  delay(10);
}
