#%%
import shutil
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

from tqdm import tqdm
import os
import argparse

#%%
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
    ap.add_argument("--c", type=float, default=1)
    ap.add_argument("--n_fourier_components", type=int, default=-1)
    ap.add_argument("--sensor_type", type=str, default="sensor")

    args = ap.parse_args()
    
    train_as = np.random.choice(np.linspace(0.1, 1, 100000), args.n_ic, replace=False)
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
            c=args.c,
            sensor_type=args.sensor_type,
            f_fourier_components=args.n_fourier_components,
            )


def generate_simulation(root, i, ic_func, sensors, x0, x1, t1, n_t, n_x, c, n_fourier_components=-1, sensor_type='sensor'):
    y, x, t = gen_wave_data_ivp(
        c=c, x0=x0, x1=x1, t1=t1, n_t=n_t, n_x=n_x, ic=ic_func
    )
    assert not (n_fourier_components == -1 and sensor_type == 'fourier'), 'n_fourier_components must be specified for fourier sensor'
    if sensor_type == 'sensor':
        u = sense_func(ic_func, sensors)
    elif sensor_type == 'fourier':
        u = sense_fourier(ic_func, sensors, 10)
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

def sense_fourier(func, points, num_components):
    y = func(points)
    fft = np.fft.fft(y)
    fft[num_components+1:-num_components] = 0

    fft0 = fft[:n_components+1]
    fft1 = fft[-n_components-1:]
    fft_ = np.concatenate((fft0, fft1))
    fft_ = np.array([[i.real, i.imag] for i in fft_])
    fft_ = fft_.reshape(-1)
    return fft_

def ic_sin(a, b):
    return lambda x: np.sin(a * x + b)

#%%
if __name__ == "__main__":
    # generate_simulation()
    generate_dataset()


#%%
ic = ic_sin(1, 1)

sensed = sense_func(ic, np.linspace(-np.pi, np.pi, 100))
sensed.shape

#%%

ic = ic_sin(1, 1)

n_points = 1000
n_components = 10
x = np.linspace(-10*np.pi, 10*np.pi, n_points)
y = ic(x)
plt.plot(x, y)
plt.show()

fft = np.fft.fft(y)
plt.plot(np.abs(fft))
plt.show()

# zero out all but the first n_components
fft[n_components+1:-n_components] = 0

# inverse fourier transform
y_ = np.fft.ifft(fft)
plt.plot(x, y, label='original')
plt.plot(x, y_, label='reconstructed')

plt.show()

#%%
def sense_fourier(func, points, num_components):
    y = func(points)
    fft = np.fft.fft(y)
    fft[num_components+1:-num_components] = 0

    fft0 = fft[:n_components+1]
    fft1 = fft[-n_components-1:]
    fft_ = np.concatenate((fft0, fft1))
    fft_ = np.array([[i.real, i.imag] for i in fft_])
    fft_ = fft_.reshape(-1)
    return fft_

ic = ic_sin(1, 1)

n_points = 1000
n_components = 10
x = np.linspace(-10*np.pi, 10*np.pi, n_points)

fft = sense_fourier(ic, x, n_components)
print(fft.shape)
print(fft)
fft_recovered = np.zeros_like(x, dtype=np.complex128)
fft_recovered[:n_components+1] = fft[:n_components+1]# fft0
fft_recovered[-n_components-1:] = fft[-n_components-1:] # fft1

y_ = np.fft.ifft(fft_recovered)
plt.plot(x, y, label='original')
plt.plot(x, y_, label='reconstructed')
plt.legend()
plt.show()


#%%
ic = ic_sin(1, 1)

n_points = 1000
n_components = 10
x = np.linspace(-10*np.pi, 10*np.pi, n_points)
y = ic(x)
plt.plot(x, y)
plt.show()

fft = np.fft.fft(y)
plt.plot(np.abs(fft))
plt.show()

# zero out all but the first n_components
fft[n_components+1:-n_components] = 0
fft0 = fft[:n_components+1]
fft1 = fft[-n_components-1:]

fft_ = np.concatenate((fft0, fft1))

fft_recovered = np.zeros_like(fft)
fft_recovered[:n_components+1] = fft_[:n_components+1]# fft0
fft_recovered[-n_components-1:] = fft_[-n_components-1:] # fft1

print(fft_recovered.shape)
print(fft_recovered.dtype)
# inverse fourier transform
y_ = np.fft.ifft(fft_recovered)
plt.plot(x, y, label='original')
plt.plot(x, y_, label='reconstructed')

plt.show()

# %%

#%%
print(fft.shape)

# %%

ic = ic_sin(1, 2)

n_points = 1000
n_components = 20
x = np.linspace(-10*np.pi, 10*np.pi, n_points)
y = ic(x)
plt.plot(x, y)
plt.show()

fft = np.fft.fft(y)
plt.plot(np.abs(fft))
plt.show()

# zero out all but the first n_components
fft[n_components:] = 0
for i in range(n_components):
    fft[-i] = fft[i]
# inverse fourier transform
y_ = np.fft.ifft(fft)
plt.plot(x, y, label='original')
plt.plot(x, y_, label='reconstructed')
plt.show()

# %%
def sense_fourier(func: Callable, points, n_components) -> np.ndarray:
    y = func(points)
    fft = np.fft.fft(y)
    return fft[:n_components]
    return np.concatenate([fft[:n_components+1], fft[-n_components:]])
#%%
a = 1
b = 1
num_points = 10000
num_components = 10

ic = ic_sin(a, b)
points = np.linspace(-np.pi, np.pi, num_points)
t = ic(points)
fft = sense_fourier(ic, points, num_components)

_ = np.zeros(len(points))
for i in range(num_components):
    _[i] = fft[i]
    _[-i] = fft[i]
y = np.fft.ifft(_)
plt.plot(points, t, label='true')
plt.plot(points, y, label='reconstructed')
plt.legend()
plt.show()
# %%
