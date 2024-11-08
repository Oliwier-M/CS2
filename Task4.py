import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def dx_dt(x, y):
    return x * (3 - x - 2 * y)

def dy_dt(x, y):
    return y * (2 - x - y)


def midpoint_method(x, y, dt):
    kx = dt * dx_dt(x, y)
    ky = dt * dy_dt(x, y)

    x_mid = x + kx / 2
    y_mid = y + ky / 2

    x_next = x + dt * dx_dt(x_mid, y_mid)
    y_next = y + dt * dy_dt(x_mid, y_mid)
    return x_next, y_next

def simulate_phase_portrait(x_range, y_range, dt=0.01, t_max=10.0):
    # Set up a grid of initial conditions
    x_initial = np.linspace(*x_range, 5)
    y_initial = np.linspace(*y_range, 5)

    plt.figure(figsize=(10, 10))

    for x0 in x_initial:
        for y0 in y_initial:
            # Initialize trajectory lists
            x_vals, y_vals, colors = [x0], [y0], [0]
            x, y = x0, y0
            time = 0

            # Simulate trajectory over time
            while time < t_max:
                x, y = midpoint_method(x, y, dt)
                x_vals.append(x)
                y_vals.append(y)
                colors.append(time)

                # Stop if trajectory goes out of bounds
                if x ** 2 + y ** 2 > 100:
                    break

                time += dt

            # Plot the trajectory with color changing over time
            plt.scatter(x_vals, y_vals, c=colors, cmap='viridis', s=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Phase Portrait with Time-Progressive Color")
    plt.colorbar(label="Time")
    plt.grid()
    plt.show()


# Finding fixed points by solving dx_dt = 0 and dy_dt = 0
def equations(vars):
    x, y = vars
    return [dx_dt(x, y), dy_dt(x, y)]


# Initial guesses for fixed points
initial_guesses = [(0, 0), (3, 0), (0, 2), (1, 1)]
fixed_points = [fsolve(equations, guess) for guess in initial_guesses]

print("Fixed Points:")
for point in fixed_points:
    print(f"({point[0]:.2f}, {point[1]:.2f})")

simulate_phase_portrait((0, 3), (0, 3))

def dx_dt_modified(x, y):
    return x * (3 - x - 1.5 * y)

def dy_dt_modified(x, y):
    return y * (2 - 0.5 * x - y)

def simulate_modified_phase_portrait(x_range, y_range, dt=0.01, t_max=10.0):
    x_initial = np.linspace(*x_range, 5)
    y_initial = np.linspace(*y_range, 5)

    plt.figure(figsize=(10, 10))

    for x0 in x_initial:
        for y0 in y_initial:
            x_vals, y_vals, colors = [x0], [y0], [0]
            x, y = x0, y0
            time = 0

            while time < t_max:
                kx = dt * dx_dt_modified(x, y)
                ky = dt * dy_dt_modified(x, y)

                x_mid = x + kx / 2
                y_mid = y + ky / 2

                x_next = x + dt * dx_dt_modified(x_mid, y_mid)
                y_next = y + dt * dy_dt_modified(x_mid, y_mid)

                x, y = x_next, y_next
                x_vals.append(x)
                y_vals.append(y)
                colors.append(time)

                if x ** 2 + y ** 2 > 100:
                    break

                time += dt

            plt.scatter(x_vals, y_vals, c=colors, cmap='plasma', s=1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Modified Phase Portrait for Stable Lotka-Volterra")
    plt.colorbar(label="Time")
    plt.grid()
    plt.show()

simulate_modified_phase_portrait((0, 3), (0, 3))
