import numpy as np
import matplotlib.pyplot as plt

def system1_fx(x, y):
    return y

def system1_fy(x, y):
    return -x

def system2_fx(x, y):
    return y

def system2_fy(x, y):
    return -np.sin(x)

def system3_fx(x, y):
    return y

def system3_fy(x, y):
    if abs(x) > 1e3:  # Prevent overflow by clamping x to a reasonable limit
        x = np.sign(x) * 1e3
    return -x + x**3

def system4_fx(x, y):
    return y

def system4_fy(x, y):
    return x - x**3

# Midpoint method
def midpoint_method(fx, fy, x0, y0, t_max, dt):
    t = np.arange(0, t_max, dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0], y[0] = x0, y0

    for i in range(1, len(t)):
        kx1 = dt * fx(x[i - 1], y[i - 1])
        ky1 = dt * fy(x[i - 1], y[i - 1])

        kx2 = dt * fx(x[i - 1] + kx1 / 2, y[i - 1] + ky1 / 2)
        ky2 = dt * fy(x[i - 1] + kx1 / 2, y[i - 1] + ky1 / 2)

        x[i] = x[i - 1] + kx2
        y[i] = y[i - 1] + ky2

    return x, y

# find isoclines
def find_isoclines(fx, fy, x_range, y_range):
    x_vals = np.linspace(*x_range, 1000)
    y_vals = np.linspace(*y_range, 1000)

    # Collecting x-isoclines (fx = 0 -> y=0)
    isoclines_x = [0] * len(x_vals)

    # Collecting y-isoclines (fy = 0)
    isoclines_y = []
    for x in x_vals:
        if np.isclose(fy(x, 0), 0, atol=1e-2):  # Check if fy(x, 0) is approximately 0
            isoclines_y.append(x)

    return x_vals, isoclines_x, isoclines_y

def plot_phase_portrait_with_isoclines(ax, fx, fy, x_range, y_range, dt, t_max, title):
    for x0 in np.linspace(*x_range, 10):
        for y0 in np.linspace(*y_range, 10):
            x, y = midpoint_method(fx, fy, x0, y0, t_max, dt)
            ax.plot(x, y, lw=0.7)

    # Find isoclines
    x_vals, isoclines_x, isoclines_y = find_isoclines(fx, fy, x_range, y_range)

    # Plot horizontal isocline for fx = 0 (y = 0)
    ax.axhline(0, color='blue', lw=1.5, label='Isocline $f_x = 0$ (y=0)')

    # Plot vertical isoclines for fy = 0
    for x_iso in isoclines_y:
        ax.axvline(x_iso, color='red', lw=1.5, linestyle='--', label='Isocline $f_y = 0$')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

# Parameters for simulation
dt = 0.01
t_max = 20

# System definitions
systems = [
    (system1_fx, system1_fy, (-2, 2), (-2, 2), dt, t_max, 'Phase Portrait: \u00a8x + x = 0'),
    (system2_fx, system2_fy, (-2 * np.pi, 2 * np.pi), (-2, 2), dt, t_max, 'Phase Portrait: \u00a8x + sin(x) = 0'),
    (system3_fx, system3_fy, (-2, 2), (-2, 2), dt, t_max, 'Phase Portrait: \u00a8x = -x + x^3'),
    (system4_fx, system4_fy, (-2, 2), (-2, 2), dt, t_max, 'Phase Portrait: \u00a8x = x - x^3')
]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Run and plot phase portraits with isoclines for each system
for i, (fx, fy, x_range, y_range, dt, t_max, title) in enumerate(systems):
    plot_phase_portrait_with_isoclines(axs[i], fx, fy, x_range, y_range, dt, t_max, title)

plt.tight_layout()
plt.show()
