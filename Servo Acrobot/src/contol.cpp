// #include <Arduino.h>
// #include <Servo.h>
// #include <Encoder.h>

// #define SERVO_PIN 9
// #define ZERO 1485
// #define MICROS_PER_PI 1310

// #define ENCODER_CPR 8192

// Servo servo;
// Encoder encoder(16, 15);

// void setup() {
//   Serial.begin(115200);
//   servo.attach(SERVO_PIN);
//   servo.writeMicroseconds(ZERO);
//   while (!Serial.available()) {}
//   encoder.write(ENCODER_CPR / 2);
// }

// void loop() {
//   if (Serial.available()) {
//     float servo_pos, encoder_pos;
//     Serial.readBytes((char*) &servo_pos, sizeof(servo_pos));
//     int servo_micros = ZERO + servo_pos / M_PI * MICROS_PER_PI;
//     servo.writeMicroseconds(servo_micros);

//     encoder_pos = encoder.read() * 2 * M_PI / ENCODER_CPR; 
//     Serial.write((char*) &encoder_pos, sizeof(encoder_pos));
//   }
// }
