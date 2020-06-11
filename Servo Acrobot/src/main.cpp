#include <Arduino.h>
#include <Servo.h>
#include <Encoder.h>

#define SERVO_PIN 9
#define ZERO 1485
#define MICROS_PER_PI 1310

#define ENCODER_CPR 8192

Servo servo;
Encoder encoder(15, 16);

unsigned long prev_time = 0;

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN);
}

void loop() {

  servo.writeMicroseconds(ZERO);
  while (!Serial.available()) {}
  float servo_pos;
  Serial.readBytes((char*) &servo_pos, sizeof(servo_pos));
  int servo_micros = ZERO + servo_pos / M_PI * MICROS_PER_PI;
  encoder.write(0);
  servo.writeMicroseconds(servo_micros);

  for (size_t i = 0; i < 1000;) {
  unsigned long curr_time = micros();
    if (curr_time - prev_time >= 1000) {
        prev_time = curr_time;

      float encoder_pos = encoder.read() * 2 * M_PI / ENCODER_CPR; 
      Serial.write((char*) &encoder_pos, sizeof(encoder_pos));
      i++;
    }
  }

}
