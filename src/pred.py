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
        for i, xt in tqdm(enumerate(trunk_input)):
            deep_output[i] = model(branch_input.unsqueeze(0), xt.unsqueeze(0)).item()
    deep_output = deep_output.reshape(ys.shape)

    return deep_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/work3/s216416/deeponet/data/1a/100.npy")
    parser.add_argument("--model_path", type=str, default='models/1a/DON_Dense_100_128_128.pt')
    parser.add_argument("--out_name", default="tmp.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = np.load(args.data, allow_pickle=True)
   
    true_y = data['y'][0]

    model = DeepONet(n_sensors=100, n_hidden=128, n_output=128)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    model.to('cpu')
    print('trajectorizzing')
    trajectories = predict_trajectories(model, data)
    print("ttrajectorzided")

    error = np.abs(true_y - trajectories)
    mae = np.mean(error)

    # Plot trajectories
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(true_y)
    ax[0].set_title('True')
    ax[1].imshow(trajectories)
    ax[1].set_title('Predicted')
    ax[2].imshow(error)
    ax[2].set_title('MAE: {}'.format(mae))
    plt.savefig(os.path.join('figs', args.out_name))
