#include <Arduino.h>

#define SERVO_PIN 3

void setup() {
  Serial.begin(115200);
  pinMode(SERVO_PIN, OUTPUT);
  analogWrite(SERVO_PIN, 128);
}

void loop() {
  Serial.println("Switch");
  analogWrite(SERVO_PIN, 96);
  delay(500);
  Serial.println("Switch");
  analogWrite(SERVO_PIN, 160);
  delay(500);
}