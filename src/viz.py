import numpy as np
import matplotlib.pyplot as plt
import argparse


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
    xx, tt = np.meshgrid(x, t)
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, tt, u, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("u")
    # colorbar
    cbar = fig.colorbar(ax.pcolormesh(xx, tt, u, shading="auto"))
    cbar.set_label("u")
    plt.show()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/0.npy")
    parser.add_argument("--mode", type=str, default="static")
    args = parser.parse_args()
    data = np.load(args.data, allow_pickle=True)
    x, t, y, u = data[0]
    if args.mode == "animate":
        viz_animate(y, x, t)
    elif args.mode == "static":
        viz_static(y, x, t)
    else:
        raise ValueError("Invalid mode")