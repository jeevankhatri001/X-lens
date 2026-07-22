/*
  XIAO ESP32-S3 Sense -- BUILT-IN mic test
  ------------------------------------------
  Tests the onboard PDM microphone that ships with the Sense board.
  No external wiring needed -- this uses the mic already built into
  the board itself.

  Requires esp32 core 3.x (ships with ESP_I2S).
*/

#include <ESP_I2S.h>

I2SClass I2S;

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("==== XIAO Sense BUILT-IN mic test ====");

  // XIAO ESP32-S3 Sense onboard mic PDM pins
  I2S.setPinsPdmRx(42, 41);   // SCK=GPIO42, SD=GPIO41 (onboard mic)

  if (!I2S.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Built-in mic init failed!");
    while (true) delay(1000);
  }

  Serial.println("Built-in mic ready. Listening -- talk or tap near the board.");
}

void loop() {
  long peak = 0;
  int count = 0;
  unsigned long start = millis();

  while (millis() - start < 300) {
    int sample = I2S.read();
    if (sample != 0 && sample != -1) {
      int v = abs(sample);
      if (v > peak) peak = v;
      count++;
    }
  }

  Serial.printf("samples=%4d   peak amplitude=%6ld   %s\n",
                count, peak,
                peak > 500 ? "<-- sound detected" : "");

  delay(700);
}
