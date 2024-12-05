import numpy as np
import multiprocessing
import csv
from tqdm import tqdm
# import psutil
# import time


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
def rk45_step(func, t, y, dt):
    A = [0,2/9,1/3,3/4,1,5/6]
    B = [[0,0,0,0,0],[2/9,0,0,0,0],[1/12,1/4,0,0,0],[69/128,-243/128, 135/64,0,0],[-17/12,27/4,-27/5,16/15,0],[65/432, -5/16,13/16,4/27,5/144]]
    B = np.array(B)
    C = [1/9,0,9/20, 16/45, 1/12]
    CH = [47/450,0,12/25,32/225,1/30,6/25]
    CT = [1/150, 0, -3/100, 16/75, 1/20, -6/25]

    k1 = dt* func(t + A[0] *dt, y)
    k2 = dt*func(t + A[1] * dt, y + B[1,0] * k1)
    k3 = dt*func(t + A[2] * dt, y + B[2,0] * k1 + B[2,1]* k2)
    k4 = dt*func(t + A[3] * dt, y + B[3,0] * k1 + B[3,1]* k2 + B[3,2] *k3)
    k5 = dt* func(t + A[4] * dt, y + B[4,0] * k1 + B[4,1]* k2 + B[4,2] *k3+ B[4,3]*k4)
    k6 = dt * func(t + A[5] * dt, y + B[5,0] * k1 + B[5,1]* k2 + B[5,2] *k3+ B[5,3]*k4+ B[5,4]*k5)
    ks = [k1,k2,k3,k4,k5,k6]
    y_new = y
    for i in range(6):
        y_new = y_new +  CH[i] * ks[i]
    TE = 0
    for i in range(6):
        TE += CT[i] * ks[i]
    return y_new, np.linalg.norm(TE)


def rk4_adaptive_step(f, t, y, dt, E_origin):
    tolerance = 1e-6

    y_full,TE = rk45_step(f, t, y, dt)

    #y_half = rk45_step(f, t + dt / 2, y, dt / 2)

    # Compute error
    #error = np.linalg.norm(y_full - y_half)
    E_after = compute_energy(y_full)
    dE = abs(E_origin - E_after)
    #dE = 0
    # Adjust step size
    
    if TE > tolerance or dE > 0.01:
        dt_new = 0.9 * dt * (tolerance/TE)**(1/5)
        return rk4_adaptive_step(f, t, y, dt_new,E_origin)  # Retry with smaller dt
    else:
        return y_full, dt


def gen_energy_adaptive(y0, i):
    global dt
    t0 = 0.0  # initial time

    # Simulation loop
    y0 = np.array(y0).astype(np.longdouble)
    t_values = [t0]
    y_values = [y0]
    ener = [compute_energy(y0)]
    y = y0
    t = t0

    E_origin = compute_energy(y)
    pbar = tqdm(total=999, desc=f"Overall Progress{i}", position=i)

    while t < t_end:
        # temp = psutil.sensors_temperatures()["coretemp"][0].current
        # while temp > 85:
        #    time.sleep(0.2)
        #    temp = psutil.sensors_temperatures()["coretemp"][0].current

        y, dt_waste = rk4_adaptive_step(equations_of_motion, t, y, dt, E_origin)
        if abs(compute_energy(y)-E_origin) > 2:
            print("helpt")
        t_old = int(t * 100)
        t += dt_waste
        if t_old < int(t * 100):
            # print(t)
            pbar.update(1)
        t_values.append(t)
        y_values.append(y)
        ener.append(compute_energy(y))

    # Convert results to numpy array for easier access
    y_values = np.array(y_values)

    # Results
    phi1 = y_values[:, 0]
    phi2 = y_values[:, 2]
    v1 = y_values[:, 1]
    v2 = y_values[:, 3]
    
    return [phi1, v1, phi2, v2, t_values, ener]

def correct_y(y, w):
    # correct the new y value by a factor to scale the energy post rk4
    phi1, v1, phi2, v2 = y
    return [phi1, w * v1, phi2, w * v2]


def gen_y0(num):
    # generate y0 values randomly, always with 0 velocity start
    y0s = []
    for i in range(num):
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)
        y0s.append([phi1, 0, phi2, 0])
    return y0s


def gen_test(args):
    y0, i = args
    with open(f"testfiles/test{i}.csv", "w", newline='') as file:
        result = []
        writer = csv.writer(file)
        phi1, v1, phi2, v2, t, ener = gen_energy_adaptive(y0, i)
        writer.writerow(["t", "phi1", "v1", "phi2", "v2", "energy"])
        for i in range(0, len(phi1)):
            entry = [t[i], phi1[i], v1[i], phi2[i], v2[i], ener[i]]
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


def comp_potential(y, g=9.81):
    # Computes the Potential Energy

    phi1, v1, phi2, v2 = y
    potential = -g * (np.cos(phi1) + np.cos(phi1) + np.cos(phi2))
    return potential

def main():
    L1 = 1
    L2 = 1
    m1 = 1
    m2 = 1
    g = 9.81
    t_end = 100
    dt = 0.001

    y0 = [np.pi - 0.1, 0, np.pi - 3.4, 0]
    y0s = gen_y0(100)

    args = [(i, count) for count, i in enumerate(y0s)]

    num_processes = 10 #multiprocessing.cpu_count() - 2
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(gen_test, args)
    result = np.array(gen_test((y0, 100)))
    print("finisher")
    energy = np.array(list(map(compute_energy, result[:,1:5])))
    print(energy.mean())
    print(energy.std())
L1 = 1
L2 = 1
m1 = 1
m2 = 1
g = 9.81
t_end = 10
dt = 0.001


main()
