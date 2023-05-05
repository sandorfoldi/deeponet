import shutil
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

from tqdm import tqdm
import os
import argparse

def generate_dataset():

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/default")
    ap.add_argument("--n_ic", type=int, default=1000)
    args = ap.parse_args()

    root = args.root
    n_ic = args.n_ic
    
    train_as = np.random.choice(np.linspace(0.1, 10, 100000), n_ic, replace=False)
    train_bs = np.random.choice(np.linspace(-np.pi, np.pi, 100000), n_ic, replace=False)
    x0 = -np.pi
    x1 = np.pi
    t1 = 100
    dt = 1
    dx = 0.063
    c = 1
    sensors = np.linspace(-np.pi, np.pi, 10)

    i = list(range(n_ic))

    if os.path.exists(root):
        r = input(f'{root} exists, delete? (y/n)')
        if r == 'y':
            shutil.rmtree(root)
        else:
            exit(0)
    os.makedirs(root, exist_ok=False)

    for a, b, i in tqdm(zip(train_as, train_bs, i)):
        generate_simulation(
            root=root,
            i=i, 
            ic_func=ic_sin(a, b), 
            sensors=sensors,
            x0=x0,
            x1=x1,
            t1=t1,
            dt=dt,
            dx=dx,
            c=c
            )


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


def sense_func(func: Callable, sensors: np.ndarray) -> np.ndarray:
    return func(sensors)


def ic_sin(a, b):
    return lambda x: np.sin(a * x + b)

def generate_simulation(root, i, ic_func, sensors, x0, x1, t1, dt, dx, c):
    y, x, t = gen_wave_data_ivp(
        c=1, x0=x0, x1=x1, t1=t1, dt=dt, dx=dx, ic=ic_func
    )

    u = sense_func(ic_func, sensors)
    data = np.array(
        [(x, t, y, u)],
        dtype=[
            ("x", np.ndarray),
            ("t", np.ndarray),
            ("y", np.ndarray),
            ("u", np.ndarray),
        ],
    )
    np.save(f"{os.path.join(root, str(i))}.npy", data)


if __name__ == "__main__":
    # generate_simulation()
    generate_dataset()