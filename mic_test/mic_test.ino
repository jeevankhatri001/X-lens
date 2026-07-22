/*
  XIAO ESP32-S3 -- external PDM mic test
  ------------------------------------------
  Wiring:
    Mic VDD -> 3V3
    Mic GND -> GND
    Mic SCK -> D8
    Mic SD  -> D9
    Mic LR  -> GND (selects left channel)
    Mic HS  -> left unconnected

  Reads audio samples for a short window and prints the peak
  amplitude. Talk/tap near the mic while it's listening --
  peak should jump noticeably compared to silence.

  Requires esp32 core 3.x (ships with ESP_I2S).
*/

#include <ESP_I2S.h>

I2SClass I2S;

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("==== External PDM mic test ====");

  // PDM RX: clock pin, data pin
  I2S.setPinsPdmRx(8, 9);   // SCK=D8, SD=D9

  if (!I2S.begin(I2S_MODE_PDM_RX, 16000, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("PDM mic init failed! Check wiring.");
    while (true) delay(1000);
  }

  Serial.println("Mic ready. Listening every second -- talk or tap near it.");
}

void loop() {
  long peak = 0;
  int count = 0;
  unsigned long start = millis();

  // sample for 300ms and track the loudest sample seen
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

  delay(700);  // total ~1s per loop
}
