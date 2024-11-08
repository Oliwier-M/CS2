import numpy as np
from matplotlib import pyplot as plt

def linear_system_fx(x, y, a, b):
    return a * x + b * y

def linear_system_fy(x, y, c, d):
    return c * x + d * y

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

matrices = [
    (np.array([[-2, 1], [0, 2]]), 'Matrix a'),
    (np.array([[3, -4], [2, -1]]), 'Matrix b'),
    (np.array([[-3, -2], [-1, -3]]), 'Matrix c'),
    (np.array([[2, 0], [0, 2]]), 'Matrix d')
]

def find_isoclines(fx, fy, x_range, y_range):
    x_vals = np.linspace(*x_range, 1000)
    y_vals = np.linspace(*y_range, 1000)

    # Collecting x-isoclines (fx = 0, -> y=0)
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

    # Find and plot isoclines
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

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (matrix, label) in enumerate(matrices):
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    def fx(x, y):
        return linear_system_fx(x, y, a, b)

    def fy(x, y):
        return linear_system_fy(x, y, c, d)

    # Plot the phase portrait on the specified subplot
    plot_phase_portrait_with_isoclines(axs[i], fx, fy, (-5, 5), (-5, 5), dt, t_max, f'Phase Portrait for {label}')

    # Trace and determinant
    trace = a + d
    determinant = a * d - b * c

    # Eigenvaleu calculation using the quadratic formula
    discriminant = trace ** 2 - 4 * determinant
    if discriminant >= 0:
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
    else:
        lambda1 = (trace + 1j * np.sqrt(-discriminant)) / 2
        lambda2 = (trace - 1j * np.sqrt(-discriminant)) / 2

    print(f"{label}: Trace = {trace}, Determinant = {determinant}")
    print(f"Eigenvalues: λ1 = {lambda1}, λ2 = {lambda2}")

    # Classification logic based on eigenvalues
    if np.isreal(lambda1) and np.isreal(lambda2):  # Both eigenvalues are real
        if lambda1 > 0 and lambda2 > 0:
            print("Classification: Unstable Node")
        elif lambda1 < 0 and lambda2 < 0:
            print("Classification: Stable Node")
        else:
            print("Classification: Saddle Point")
    else:  # Complex eigenvalues
        if np.real(lambda1) > 0:  # Unstable spiral
            print("Classification: Unstable Spiral")
        elif np.real(lambda1) < 0:  # Stable spiral
            print("Classification: Stable Spiral")
        else:  # Purely imaginary eigenvalues
            print("Classification: Center (Degenerate)")

plt.tight_layout()
plt.show()
