import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
from scipy.integrate import solve_ivp
from tqdm import tqdm
from typing import Callable, List, Tuple
import pandas as pd
from utils import load_obj
import shutil


def sense_func(func: Callable, sensors: np.ndarray) -> np.ndarray:
    return func(sensors)


def gen_wave_data_ivp(
    c: float, x0: float, x1: float, t1: float, n_t: float, n_x: float, ic: Callable
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
    v0 = np.zeros(len(x))
    y0 = np.concatenate((u0, v0))

    def wave_equation(t, y, c):
        u, v = np.split(y, 2)
        d2udx2 = np.gradient(np.gradient(u))
        return np.concatenate((v, c**2 * d2udx2))

    sol = solve_ivp(lambda t, y: wave_equation(t, y, c), [0, t1], y0, t_eval=t)
    u, v = np.split(sol.y, 2)
    return u, x, t


def generate_simulation(path, ic_func, x0, x1, t1, nt, nx, c,):
    y, x, t = gen_wave_data_ivp(
        c=c, x0=x0, x1=x1, t1=t1, n_t=nt, n_x=nx, ic=ic_func
    )

    data = np.array(
        [(x, t, y)],
        dtype=[
            ("x", np.ndarray),
            ("t", np.ndarray),
            ("y", np.ndarray),
        ],
    )
    np.save(path, data)


def main(args):
    """
    args:
    x0
    x1
    dx
    dt
    nt

    icf
    fdim
    fmin
    fmax
    fn

    c
    """
    # hyperparameters:
    # # dimensionality and grid spacing
    # # input space definition as in mapping between grid point vector and initial condition
    # # wave equation parameters
    # # observation range, e.g. xmin xmax dx t1 dt

    # steps:
    # generate n dimensinal equidistant grid
    # flatten grid
    # for every row, generate wave data
    # save wave data and save metadata in csv file

    # generate grid
    simdir = os.path.join(args.root, 'sim')
    f_axes = [np.linspace(args.fmin, args.fmax, args.fn) for i in range(args.fdim)]
    grid = np.array(np.meshgrid(*f_axes))
    grid = grid.reshape(args.fdim, -1).T

    print(f'grid shape: {grid.shape}')

    ic_func = load_obj(args.icf)

    if not os.path.exists(args.root):
        os.mkdir(args.root)
        os.mkdir(simdir)
    else:
        _ = input("Root path already exists. Overwrite it? (y) ")
        if _ == 'y':
            shutil.rmtree(args.root)
            os.mkdir(args.root)
            os.mkdir(simdir)
    metadata_csv = []
    paths_csv = []
    # metadata table columns:
    # # initial condition function name
    # # grid point vector
    # # wave equation parameters
    # # observation range, e.g. xmin xmax dx t1 dt
    # # num sensors
    # # path
    metadata_csv = [{
        'icf': args.icf,
        'c': args.c,
        'x0': args.x0,
        'x1': args.x1,
        't1': args.t1,
        'nt': args.nt,
        'nx': args.nx,
    }]

    for i, point in tqdm(enumerate(grid)):
        path = os.path.join(args.root, 'sim', str(i) + '.npy')

        generate_simulation(
            path=path,
            ic_func=ic_func(point),
            x0=args.x0,
            x1=args.x1,
            t1=args.t1,
            nt=args.nt,
            nx=args.nx,
            c=args.c,
        )
        paths_csv.append(
            [path, *point]
        )
    
    metadata_csv = pd.DataFrame(metadata_csv)
    metadata_csv.to_csv(os.path.join(args.root, 'metadata.csv'))

    paths_csv = pd.DataFrame(paths_csv)
    paths_csv.to_csv(os.path.join(args.root, 'paths.csv'))




if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--root', type=str, default='data/wave_generator')

    ap.add_argument('--c', type=float, default=1)

    ap.add_argument('--x0', type=float, default=0)
    ap.add_argument('--x1', type=float, default=1)
    ap.add_argument('--dx', type=float, default=0.1)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--nt', type=int, default=10)

    ap.add_argument('--icf', type=str, default='data/wave_generator/icf/linear_icf.pkl')
    ap.add_argument('--fdim', type=int, default=3)
    ap.add_argument('--fmin', type=float, default=0)
    ap.add_argument('--fmax', type=float, default=100)
    ap.add_argument('--fn', type=int, default=100)

    args = ap.parse_args()

    args.nx = int((args.x1 - args.x0) / args.dx)
    args.t1 = args.nt * args.dt

    print('args loaded')

    main(args)
