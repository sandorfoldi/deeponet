import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import torch
from src.model import DeepONet
from tqdm import tqdm


# Predict trajectories for 1000 intial conditions
def predict_trajectories(model, dataset):
    # Load the dataset
    xs = dataset['x'][0]
    ts = dataset['t'][0]
    ys = dataset['y'][0]
    us = dataset['u'][0]

    XT = np.meshgrid(xs, ts)
    branch_input = torch.from_numpy(us).float()
    trunk_input = torch.from_numpy(np.vstack([XT[0].ravel(), XT[1].ravel()]).T).float()

    model.eval()
    deep_output = np.zeros_like(ys).flatten()

    with torch.no_grad():    
        for i, xt in tqdm(enumerate(trunk_input)):
            deep_output[i] = model(branch_input.unsqueeze(0), xt.unsqueeze(0)).item()
    deep_output = deep_output.reshape(ys.shape)

    return deep_output


# Predict trajectories for 1000 intial conditions
def predict_trajectories_batched(model, dataset):
    # Load the dataset
    xs = dataset['x'][0]
    ts = dataset['t'][0]
    ys = dataset['y'][0]
    us = dataset['u'][0]

    XT = np.meshgrid(xs, ts)
    trunk_input = torch.from_numpy(np.vstack([XT[0].ravel(), XT[1].ravel()]).T).float()
    branch_input = torch.from_numpy(us).float()

    model.eval()
    deep_output = np.zeros_like(ys).flatten()

    with torch.no_grad():    
        # for i, xt in tqdm(enumerate(trunk_input)):
        #     deep_output[i] = model(branch_input.unsqueeze(0), xt.unsqueeze(0)).item()
        deep_output = model(torch.tile(branch_input.unsqueeze(0), (trunk_input.shape[0], 1)), trunk_input).squeeze(0).numpy()
    deep_output = deep_output.reshape((ys.shape[1], ys.shape[0])).T

    return deep_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sin_1000/100.npy")
    parser.add_argument("--model_path", type=str, default='models/sin_1000/DON_Dense_100_128_128.pt')
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--out_name", default="sin_1000.png")
    parser.add_argument("--num_sensors", type=int, default=100)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = np.load(args.data, allow_pickle=True)
   
    true_y = data['y'][0]

    model = DeepONet(n_sensors=args.num_sensors, n_hidden=args.n_hidden, n_output=args.n_hidden)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    model.to('cpu')
    print('trajectorizing')
    # trajectories = predict_trajectories(model, data)
    trajectories = predict_trajectories_batched(model, data)
    print("done")

    error = np.abs(true_y - trajectories)
    mae = np.mean(error)

    # Plot trajectories
    # put colorbar on all axes
    frac = 0.09
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the first image
    # im1 = ax[0].imshow(true_y, cmap='viridis', vmin=-.1, vmax=.1)
    im1 = ax[0].imshow(true_y, cmap='viridis')
    ax[0].set_title('True')
    fig.colorbar(im1, ax=ax[0], orientation='vertical', fraction=frac)

    # im2 = ax[1].imshow(trajectories, cmap='viridis', vmin=-.1, vmax=.1)
    im2 = ax[1].imshow(trajectories, cmap='viridis')
    ax[1].set_title('Predicted')
    fig.colorbar(im2, ax=ax[1], orientation='vertical', fraction=frac)

    im3 = ax[2].imshow(((true_y-trajectories)**2)**.5, cmap='viridis', vmin=0)
    ax[2].set_title(f'MSE: {mae:.3f}')
    fig.colorbar(im3, ax=ax[2], orientation='vertical', fraction=frac)

    fig.tight_layout()

    plt.savefig(os.path.join('figs', args.out_name))