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
    ap.add_argument("--n_t", type=int, default=100)
    ap.add_argument("--n_x", type=int, default=100)
    ap.add_argument("--n_sensors", type=int, default=100)
    ap.add_argument("--x0", type=float, default=-np.pi)
    ap.add_argument("--x1", type=float, default=np.pi)
    ap.add_argument("--t1", type=float, default=100)

    args = ap.parse_args()
    
    train_as = np.random.choice(np.linspace(0.1, 10, 100000), args.n_ic, replace=False)
    train_bs = np.random.choice(np.linspace(-np.pi, np.pi, 100000), args.n_ic, replace=False)
    sensors = np.linspace(-np.pi, np.pi, args.n_sensors)

    idxs = list(range(args.n_ic))

    if os.path.exists(args.root):
        r = input(f'{args.root} exists, delete? (y/n)\n')
        if r == 'y':
            shutil.rmtree(args.root)
        else:
            exit(0)
    os.makedirs(args.root, exist_ok=False)

    for a, b, i in tqdm(zip(train_as, train_bs, idxs)):
        generate_simulation(
            root=args.root,
            i=i, 
            ic_func=ic_sin(a, b), 
            sensors=sensors,
            x0=args.x0,
            x1=args.x1,
            t1=args.t1,
            n_t=args.n_t,
            n_x=args.n_x,
            c=args.c
            )


def generate_simulation(root, i, ic_func, sensors, x0, x1, t1, n_t, n_x, c):
    y, x, t = gen_wave_data_ivp(
        c=1, x0=x0, x1=x1, t1=t1, n_t=n_t, n_x=n_t, ic=ic_func
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


def gen_wave_data_ivp(
    c: float, x0: float, x1: float, t1: float, n_t: float, n_x: float, ic: Callable
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
    x = np.linspace(x0, x1, n_x)
    t = np.linspace(0, t1, n_t)
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


if __name__ == "__main__":
    # generate_simulation()
    generate_dataset()