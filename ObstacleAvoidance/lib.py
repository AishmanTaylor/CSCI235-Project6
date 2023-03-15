from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color

class SensorMotor:
    def __init__(self, ev3):
        self.ev3 = ev3
        self.left = Motor(Port.A)
        self.right = Motor(Port.D)

    def stop_all(self):
        self.left.run(0)
        self.right.run(0)

SPEED = 360

def go_forward(robot):
    robot.left.run(SPEED)
    robot.right.run(SPEED)

def spin_left(robot):
    robot.left.run(SPEED)
    robot.right.run(-SPEED)