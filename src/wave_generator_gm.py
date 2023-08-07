import shutil
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

from tqdm import tqdm
import os
import argparse


def ic_gm(mm, vv):
    return lambda x: sum([1/np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v)) for m in mm for v in vv])


def wave_equation(t, y, c):
    u, v = np.split(y, 2)
    d2udx2 = np.gradient(np.gradient(u))
    v[0] = 0
    v[-1] = 0
    return np.concatenate((v, c**2 * d2udx2))


def generate_dataset():

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="bvp")
    ap.add_argument("--root", type=str, default="data/default")
    ap.add_argument("--n_ic", type=int, default=1000)
    ap.add_argument("--d_t", type=float, default=0.1)
    ap.add_argument("--n_t", type=int, default=500)
    ap.add_argument("--d_x", type=float, default=0.1)
    ap.add_argument("--n_sensors", type=int, default=100)
    ap.add_argument("--x0", type=float, default=0)
    ap.add_argument("--x1", type=float, default=100)
    ap.add_argument("--c", type=float, default=5)

    ap.add_argument("--m_min", type=float, default=10)
    ap.add_argument("--m_max", type=float, default=90)
    ap.add_argument("--v_min", type=float, default=10)
    ap.add_argument("--v_max", type=float, default=10)
    ap.add_argument("--num_g", type=int, default=1)

    ap.add_argument("--n_fourier_components", type=int, default=-1)
    ap.add_argument("--sensor_type", type=str, default="sensor")
    ap.add_argument("--ic", type=str, default="sin")

    args = ap.parse_args()
    args.n_x = int((args.x1 - args.x0) / args.d_x)
    args.t1 = args.n_t * args.d_t

    sensors = np.linspace(args.x0, args.x1, args.n_sensors)

    idxs = list(range(args.n_ic))
    if os.path.exists(args.root):
        r = input(f'{args.root} exists, delete? (y/n)\n')
        if r == 'y':
            shutil.rmtree(args.root)
        else:
            exit(0)
    os.makedirs(args.root, exist_ok=False)
    if args.sensor_type == 'fourier':
        assert args.n_fourier_components % 4 == 0, 'n_fourier_components must be divisible by 4 (trunc in start and end, and complex conjugate)'
        print('[WARNING]: n_fourier_components is divided by 2 and subtracted by 2.')
        args.n_fourier_components = args.n_fourier_components/4 - 1
        print(f'[INFO]: n_fourier_components: {args.n_fourier_components}')
        args.n_fourier_components = int(args.n_fourier_components)


    for i in tqdm(idxs):
        m = [np.random.random()*(args.m_max-args.m_min) + args.m_min for _ in range(args.num_g)]
        v = [np.random.random()*(args.v_max-args.v_min) + args.v_min for _ in range(args.num_g)]

        generate_simulation(
            mode=args.mode,
            root=args.root,
            i=i, 
            ic_func=ic_gm(m, v), 
            sensors=sensors,
            x0=args.x0,
            x1=args.x1,
            n_t=args.n_t,
            t1=args.t1,
            n_x=args.n_x,
            c=args.c,
            sensor_type=args.sensor_type,
            n_fourier_components=args.n_fourier_components,
            )


def generate_simulation(mode, root, i, ic_func, sensors, x0, x1, t1, n_t, n_x, c, n_fourier_components=-1, sensor_type='sensor'):
    if mode == 'bvp':
        y, x, t = gen_wave_data_bvp(
            c=c, x0=x0, x1=x1, t1=t1, n_t=n_t, n_x=n_x, ic=ic_func
        )
    else:
        raise NotImplementedError(f'{mode} not implemented')

    assert not (n_fourier_components == -1 and sensor_type == 'fourier'), 'n_fourier_components must be specified for fourier sensor'
    if sensor_type == 'sensor':
        u = sense_func(ic_func, sensors)
    elif sensor_type == 'fourier':
        u = sense_fourier(ic_func, sensors, n_fourier_components)
    else:
        raise NotImplementedError(f'{sensor_type} not implemented')
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

def gen_wave_data_bvp(
    c: float, x0: float, x1: float, t1: float, n_t: float, n_x: float, ic: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate wave data using the wave equation
    :param c: wave speed
    :param x0: left boundary
    :param x1: right boundary
    :param t1: end time
    :param n_t: num time steps
    :param n_x: num space steps
    :param ic: initial condition

    :return: u: solutiom
    :return: x: space
    :return: t: time
    """
    x = np.linspace(x0, x1, n_x)
    t = np.linspace(0, t1, n_t)
    u0 = ic(x)
    v0 = np.zeros_like(x)
    # v0 = ic(x+1e-2)
    y0 = np.concatenate((u0, v0))



    sol = solve_ivp(lambda t, y: wave_equation(t, y, c), [0, t1], y0, t_eval=t)
    u, v = np.split(sol.y, 2)
    return u, x, t


def sense_func(func: Callable, sensors: np.ndarray) -> np.ndarray:
    return func(sensors)

def sense_fourier(func, points, num_components):
    y = func(points)
    fft = np.fft.fft(y)
    fft[num_components+1:-num_components] = 0

    fft0 = fft[:num_components+1]
    fft1 = fft[-num_components-1:]
    fft_ = np.concatenate((fft0, fft1))
    fft_ = np.array([[i.real, i.imag] for i in fft_])
    fft_ = fft_.reshape(-1)
    return fft_



if __name__ == "__main__":
    # generate_simulation()
    generate_dataset()