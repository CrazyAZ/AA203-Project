# AA203-Project
This repository contains all the code I wrote for my AA203: Optimal and Learning Based Control Project. This project explored if it is possible to balance an acrobot that is actuated by a low cost servo. Here is a brief description of what is in each folder:

- standard_acrobot: This folder contains a simulator for a standard torque controllable acrobot. There is a standard LQR balancing controller and two incomplete attempts at swingup controllers.
- servo_acrobot: This folder contains a simulator for a servo controlled acrobot. The simulator in this folder is meant to simulate the physical acrobot I built as closely as possible. It uses an extended Kalman filter for a state estimation and the modelled dynamics for the actual servo.
- real_acrobot_control: This folder contains all the python code I wrote for contolling the hardware acrobot and characterizing the servo.
- Servo Acrobot: This folder contains a Platformio project with all the code that I ran on the Teensy microcontroller for controlling the hardware acrobot and characterizing the servo.

My acrobot simulators are based on an acrobot environment for use in OpenAI Gym, which I modified for for use on my problem. The original code for the environment can be found here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
