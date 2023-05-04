from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp


def wave_eq_discovery():
    # Define the initial condition
    u0 = np.sin(np.linspace(0, 2 * np.pi, 100))
    v0 = np.zeros(100)
    y0 = np.concatenate((u0, v0))

    def wave_equation(t, y, c):
        u, v = np.split(y, 2)
        d2udx2 = np.gradient(np.gradient(u))
        return np.concatenate((v, c**2 * d2udx2))

    # Define the initial condition
    u0 = np.sin(np.linspace(0, 2 * np.pi, 100))
    v0 = np.zeros(100)
    y0 = np.concatenate((u0, v0))

    # Set the wave speed
    c = 1

    # Set the time interval and solve the equation
    sol = solve_ivp(
        lambda t, y: wave_equation(t, y, c), [0, 10], y0, t_eval=np.linspace(0, 10, 100)
    )

    # Plot the solution
    fig, ax = plt.subplots()
    for i in range(len(sol.t)):
        ax.clear()
        u, v = np.split(sol.y[:, i], 2)
        ax.plot(np.linspace(0, 2 * np.pi, 100), u)
        ax.set_ylim([-1, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"t={sol.t[i]:.2f}")
        plt.pause(0.01)


def gen_wave_data_ivp(
    c: float, x0: float, x1: float, t1: float, dt: float, dx: float, ic: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate wave data using the wave equation
    :param c: wave speed
    :param x0: left boundary
    :param x1: right boundary
    :param t1: end time
    :param dt: time step
    :param dx: space step
    :param ic: initial condition

    :return: u: solutiom
    :return: x: space
    :return: t: time
    """
    x = np.arange(x0, x1, dx)
    t = np.arange(0, t1, dt)
    u0 = ic(x)
    v0 = np.zeros(len(x))
    y0 = np.concatenate((u0, v0))

    def wave_equation(t, y, c):
        u, v = np.split(y, 2)
        d2udx2 = np.gradient(np.gradient(u))
        return np.concatenate((v, c**2 * d2udx2))

    sol = solve_ivp(lambda t, y: wave_equation(t, y, c), [0, t1], y0, t_eval=t)
    u, v = np.split(sol.y, 2)
    return u, x, t


def viz_animate(u: np.ndarray, x: np.ndarray, t: np.ndarray)->None:
    """
    Visualize the wave data using animation
    """
    umin = np.min(u)
    umax = np.max(u)

    fig, ax = plt.subplots()
    for i, t_ in enumerate(t):
        ax.clear()
        ax.plot(x, u[:, i])
        ax.set_ylim([umin, umax])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"t={t_:.2f}")
        plt.pause(0.01)
    return None

def sense_func(func: Callable, sensors: np.ndarray)->np.ndarray:
    return func(sensors)

def ic_sin(a, b):
    return lambda x: np.sin(a*x+b)

if __name__ == "__main__":
    ic_func = ic_sin(1, 0)
    sensors = np.linspace(-np.pi, np.pi, 10)
    y, x, t = gen_wave_data_ivp(
        c=1,
        x0=-np.pi,
        x1=np.pi,
        t1=100,
        dt=1,
        dx=0.063,
        ic=ic_func)
    
    u = sense_func(ic_func, sensors)
    data = np.array([(x, t, y, u)], dtype=[("x", np.ndarray), ("t", np.ndarray), ("y", np.ndarray), ("u", np.ndarray)])
    np.save("data/0.npy", data)
    # viz_animate(u, x, t)
    print(y.shape)
    plt.imshow(y)
    plt.show()
    # wave_eq_discovery()
