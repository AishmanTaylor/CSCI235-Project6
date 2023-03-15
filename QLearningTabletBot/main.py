#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

import socket
import robot, lib

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()


# Write your program here.
ev3.speaker.beep()

SERVER_IP = "172.17.3.91"
PORT = 8888

def send_message(message):
    reply = None
    try:
        sock = socket.socket()
        sock.connect((SERVER_IP, PORT))
        sock.send(message.encode())
        reply = sock.recv(1024).decode()
    except Exception as e:
        reply = str(e)
    finally:
        sock.close()
    return reply

robot = lib.SensorMotor(ev3)
send_message("knn 3 avoid")

# while True:
#     msg = send_message("classify")
#     if msg == "obstacle":
#         lib.spin_left(robot)
#     elif msg == "clear":
#         lib.go_forward(robot)
#     ev3.screen.clear()
#     ev3.screen.draw_text(0, 0, msg)
#     wait(100)

CLEAR = 0
CAMERA_CLOSE = 1
SONAR_CLOSE = 2
SHELTER = 3
# LEFT_BUMP = 1
# RIGHT_BUMP = 2
# CLOSE_OBEJCT = 3
# MID_OBJECT = 4
# FAR_OBJECT = 5

def find_state(bot):
    distance = bot.sonar.distance()
    msg = send_message("classify")
    if msg == "CameraClose":
        return CAMERA_CLOSE
    elif msg == "SonarClose":
        return SONAR_CLOSE
    elif 0 <= distance <= 200:
        return SHELTER
    elif msg == "clear":
        return CLEAR
    ev3.screen.clear()
    ev3.screen.draw_text(0, 0, msg)
    wait(100)

    # if bot.bump_left.pressed():
    #     return LEFT_BUMP
    # elif bot.bump_right.pressed():
    #     return RIGHT_BUMP
    # elif 0 <= distance <= 200:
    #     return CLOSE_OBEJCT
    # elif 200 <= distance <= 400:
    #     return MID_OBJECT
    # elif 400 <= distance <= 600:
    #     return FAR_OBJECT


def reward(bot, state, action):
    if state == SONAR_CLOSE:
        return -10
    elif state == CAMERA_CLOSE:
        return -5
    elif state == SHELTER:
        return 5
    elif action == 0:
        return 1
    else:
        return 0

params = lib.QParameters()
params.pause_ms = 500
params.actions = [robot.go_forward, robot.go_left, robot.go_right, robot.go_back]
params.num_states = 4
params.state_func = find_state
params.reward_func = reward
params.target_visits = 5
params.discount = 0.5
params.rate_constant = 10
params.max_steps = 200

ev3 = EV3Brick()

lib.run_q(robot.SensorMotor(ev3), params)