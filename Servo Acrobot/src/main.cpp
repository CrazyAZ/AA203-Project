#include <Arduino.h>
#include <Servo.h>
#include <Encoder.h>

#define SERVO_PIN 9
#define ZERO 1485
#define PI_OVER_2 655

Servo servo;
Encoder encoder(16, 15);

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN);
  servo.writeMicroseconds(ZERO);

}

void loop() {
  Serial.println(encoder.read());
  delay(100);
  // if (Serial.available()) {
  //   int pos = Serial.parseInt();
  //   servo.writeMicroseconds(pos);
  //   Serial.println(pos);
  //   Serial.read();
  // }
}