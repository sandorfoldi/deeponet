import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob


def viz_animate(y: np.ndarray, x: np.ndarray, t: np.ndarray) -> None:
    """
    Visualize the wave data using animation
    """
    umin = np.min(y)
    umax = np.max(y)

    fig, ax = plt.subplots()
    for i, t_ in enumerate(t):
        ax.clear()
        ax.plot(x, y[:, i])
        ax.set_ylim([umin, umax])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"t={t_:.2f}")
        plt.pause(0.01)
    return None

def viz_static(u: np.ndarray, x: np.ndarray, t: np.ndarray) -> None:
    """
    Visualize the wave data using animation
    """
    u = u.T  # Transpose u to match the dimensions
    xx, tt = np.meshgrid(x, t)
    fig, ax = plt.subplots()
    print(x.shape, t.shape, u.shape)
    ax.pcolormesh(xx, tt, u, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("u")
    # colorbar
    cbar = fig.colorbar(ax.pcolormesh(xx, tt, u, shading="auto"))
    cbar.set_label("u")
    plt.savefig('figs/tmp.png')
    return None

def viz_static_16(us, xs, ts) -> None:
    """
    Visualize the wave data using animation
    """

    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for u, x, t, ax in zip(us, xs, ts, axs.flatten()):
        u = u.T  # Transpose u to match the dimensions
        xx, tt = np.meshgrid(x, t)
        ax.pcolormesh(xx, tt, u, shading="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title("u")

    plt.tight_layout()
    plt.savefig('tmp/viz.png')
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/a")
    parser.add_argument("--mode", type=str, default="static_16")
    args = parser.parse_args()
    if args.mode == "animate":
        data = np.load(args.data, allow_pickle=True)
        viz_animate(data[0][2], data[0][0], data[0][1])
    elif args.mode == "static":
        data = np.load(args.data, allow_pickle=True)
        viz_static(data[0][2], data[0][0], data[0][1])
    elif args.mode == "static_16":
        data_paths = glob(os.path.join(args.data, '*.npy'))
        datas = [np.load(data_paths[i], allow_pickle=True) for i in range(16)]
        xs, ts, ys, us = [], [], [], []
        for i in range(16):
            xs.append(datas[i][0][0])
            ts.append(datas[i][0][1])
            ys.append(datas[i][0][2])
        viz_static_16(ys, xs, ts)
    else:
        raise ValueError("Invalid mode")
