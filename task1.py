import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * (x - 1) * (x - 2)

def euler_method(x0, f, dt, t_max):
    t_values = np.arange(0, t_max, dt)
    x_values = [x0]
    for t in t_values[:-1]:
        x = x_values[-1]
        x_next = x + dt * f(x)
        x_values.append(x_next)

    return t_values, x_values


dt_values = [0.01, 0.005, 0.001]
t_max = 5.0
fixed_points = [0, 1, 2]
offsets = [-0.1, 0.1]

for fp in fixed_points:

    plt.figure(figsize=(10, 6))
    for offset in offsets:
        x0 = fp + offset
        for dt in dt_values:
            t_values, x_values = euler_method(x0, f, dt, t_max)
            plt.plot(t_values, x_values, label=f'Initial x0={x0}, dt={dt}')

    y_range = 2

    plt.title(f"Stability analysis near fixed point x* = {fp}")
    plt.xlabel("Time t")
    plt.ylabel("x(t)")
    plt.ylim(fp - y_range, fp + y_range)
    plt.legend()
    plt.grid()
    plt.show()