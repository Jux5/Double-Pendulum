import numpy as np
import csv


def equations_of_motion(t, y):
    theta1, theta1_dot, theta2, theta2_dot = y

    # Common factors for simplicity
    delta = theta1 - theta2
    den1 = L1 * (m1 + m2) - L1 * m2 * np.cos(delta) ** 2
    den2 = L2 * (m1 + m2) - L2 * m2 * np.cos(delta) ** 2

    # Accelerations from the equations we derived
    theta1_ddot = (
        -L1 * m2 * np.sin(delta) * np.cos(delta) * theta1_dot**2
        - L2 * m2 * np.sin(delta) * theta2_dot**2
        - g * m1 * np.sin(theta1)
        - g * m2 * np.sin(theta1)
        + g * m2 * np.sin(theta2) * np.cos(delta)
    ) / den1

    theta2_ddot = (
        L1 * m1 * np.sin(delta) * theta1_dot**2
        + L1 * m2 * np.sin(delta) * theta1_dot**2
        + L2 * m2 * np.sin(delta) * np.cos(delta) * theta2_dot**2
        + g * (m1 + m2) * np.sin(theta1) * np.cos(delta)
        - g * (m1 + m2) * np.sin(theta2)
    ) / den2

    return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])


# Runge-Kutta 4th order method
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt * k1 / 2)
    k3 = func(t + dt / 2, y + dt * k2 / 2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_adaptive_step(f, t, y, dt, E_origin):
    tolerance = 1e-6

    y_full = rk4_step(f, t, y, dt)

    y_half = rk4_step(f, t + dt / 2, y, dt / 2)

    # Compute error
    error = np.linalg.norm(y_full - y_half)
    E_after = compute_energy(y_full)
    dE = abs(E_origin - E_after)
    # Adjust step size
    if error > tolerance or dE > 2:
        return rk4_adaptive_step(f, t, y, dt * 0.75, E_origin)  # Retry with smaller dt
    elif error < tolerance * 0.1 and dE < 0.5:
        return y_full, max(dt * 1.5, 0.005, dt * 2)
    else:
        return y_full, dt


def gen_energy_adaptive(y0):
    t0 = 0.0  # initial time

    # Simulation loop
    y0 = np.array(y0).astype(np.longdouble)
    t_values = [t0]
    y_values = [y0]
    y = y0
    t = t0

    E_origin = compute_energy(y)
    while t < t_end:
        y, dt = rk4_adaptive_step(equations_of_motion, t, y, dt, E_origin)
        E_new = compute_energy(y)
        dE = abs(E_origin - E_new)
        if abs(dE) >= 1:
            print(dE)
            KE = comp_kinetic(y)
            PE = comp_potential(y)
            scale_factor = np.sqrt((E_origin - PE) / KE)
            y = correct_y(y, scale_factor)
        print(t)
        t += dt
        t_values.append(t)
        y_values.append(y)

    # Convert results to numpy array for easier access
    y_values = np.array(y_values)

    # Results
    phi1 = y_values[:, 0]
    phi2 = y_values[:, 2]
    v1 = y_values[:, 1]
    v2 = y_values[:, 3]

    return [phi1, v1, phi2, v2, t_values]


def gen_energy(y0):
    t0 = 0.0  # initial time

    # Simulation loop
    t_values = [t0]
    y_values = [y0]
    y = y0
    t = t0

    E_origin = compute_energy(y)
    tolerance = E_origin * 0.000001
    while t < t_end:
        y = rk4_step(equations_of_motion, t, y, dt)
        E_new = compute_energy(y)
        dE = E_new - E_origin
        if abs(dE) >= tolerance:
            KE = comp_kinetic(y)
            PE = comp_potential(y)
            scale_factor = np.sqrt((E_origin - PE) / KE)
            y = correct_y(y, scale_factor)
        t += dt
        t_values.append(t)
        y_values.append(y)

    # Convert results to numpy array for easier access
    y_values = np.array(y_values)

    # Results
    phi1 = y_values[:, 0]
    phi2 = y_values[:, 2]
    v1 = y_values[:, 1]
    v2 = y_values[:, 3]
    return [phi1, v1, phi2, v2]


def correct_y(y, w):
    # correct the new y value by a factor to scale the energy post rk4
    phi1, v1, phi2, v2 = y
    return [phi1, w * v1, phi2, w * v2]


def gen(y0):
    t0 = 0.0  # initial time
    # Simulation loop
    t_values = [t0]
    y_values = [y0]
    y = y0
    t = t0

    while t < t_end:
        y = rk4_step(equations_of_motion, t, y, dt)
        t += dt
        t_values.append(t)
        y_values.append(y)

    # Convert results to numpy array for easier access
    y_values = np.array(y_values)

    # Results
    phi1 = y_values[:, 0]
    phi2 = y_values[:, 2]
    v1 = y_values[:, 1]
    v2 = y_values[:, 3]
    return [phi1, v1, phi2, v2, t_values]


def gen_y0(num):
    # generate y0 values randomly, always with 0 velocity start
    y0s = []
    for i in range(num):
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)
        y0s.append([phi1, 0, phi2, 0])
    return y0s


def gen_test(y0, i):
    with open(f"testfiles/test{i}.csv", "w") as file:
        result = []
        writer = csv.writer(file)
        phi1, v1, phi2, v2, t = gen_energy_adaptive(y0)
        writer.writerow(["t", "phi1", "v1", "phi2", "v2"])
        for i in range(0, len(phi1) - 10, 10):
            entry = [t[i], phi1[i], v1[i], phi2[i], v1[i]]
            writer.writerow(entry)
            result.append(entry)
    return result


def compute_energy(y):
    # Computes the total Energy
    kinetic = comp_kinetic(y)
    potential = comp_potential(y)
    return potential + kinetic


def comp_kinetic(y):
    # Compute the kinetic energy
    phi1, v1, phi2, v2 = y
    kinetic = 0.5 * v1**2 + 0.5 * (v1**2 + v2**2 + 2 * v1 * v2 * np.cos(phi1 - phi2))
    return kinetic


def comp_potential(y):
    # Computes the Potential Energy
    phi1, v1, phi2, v2 = y
    potential = -g * (np.cos(phi1) + np.cos(phi1) + np.cos(phi2))
    return potential


L1 = 1
L2 = 1
m1 = 1
m2 = 1
g = 9.81
t_end = 10
dt = 0.005

y0 = [np.pi - 0.1, 0, np.pi - 3.4, 0]
y0s = gen_y0(10)
count = 0
for i in y0s:
    result = gen_test(y0, count)
    count += 1
data = gen_test(y0, 50)
