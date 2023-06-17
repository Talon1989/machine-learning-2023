import numpy as np
import matplotlib.pyplot as plt


def position_x(value, x_0=0, v_0=1, tau=4):
    return x_0 + (v_0 * tau) * np.log(1 + (value / tau))


def position_y(value, y_0=0, v_ter=1):
    return y_0 + ((v_ter**2) / 9.8) * np.log(np.cosh((value * 9.8) / v_ter))


tau = 4
v_ter = 35
velocity_0 = 30
angle = 50
degrees_to_radians = (2 * np.pi) / 360
x_velocity_0 = velocity_0 * np.cos(angle * degrees_to_radians)
y_velocity_0 = velocity_0 * np.sin(angle * degrees_to_radians)


x_s = position_x(np.arange(50), x_velocity_0, tau=tau)
y_s = position_y(np.arange(50), y_velocity_0, v_ter=v_ter)
















