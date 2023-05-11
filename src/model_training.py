import torch 
import argparse
from model import DeepONet, DeepONetCNN
from wave_loader import get_wave_datasets
import numpy as np
import os
from glob import glob

"""
    Class used to train and save a DeepONet model:
    Input: 


    Output:
    
"""

def save_model(model, root, foldername):
    if not os.path.exists(root + '/models/' + foldername):
        os.makedirs(root + '/models/' + foldername)

    filename = str(model) + '.pt'
    torch.save(model.state_dict(), root + '/models/' + foldername + '/' + filename)


def save_results(losses, root, foldername, filename):
    if not os.path.exists(root + '/results/' + foldername):
        os.makedirs(root + '/results/' + foldername)

    np.save(root + '/results/' + foldername + '/' + filename, losses)

def train_model(args):
    dataset_path = args.dataset
    n_points = args.n_points

    model_name = args.model
    hidden_units = args.n_hidden
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    
    root = os.getcwd()

    paths = glob(root + "/" + dataset_path +"/" + '*.npy')

    if (len(paths) == 0):
        raise Exception('No files found in dataset folder')

    print(n_points)
    
    # Load dataset
    ds_train, ds_valid = get_wave_datasets(paths, n_points=n_points)

    # Train and validation loaders
    train_dataloader =  torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    # Model
    model = DeepONet(100, hidden_units, hidden_units) if model_name == 'FFNN' else DeepONetCNN(100, hidden_units, hidden_units) # Allow for CNN

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # Train
    epoch_train_losses, epoch_val_losses = [], []

    print(f'Training {str(model)} for {epochs} epochs')

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for (xt_batch, y_batch, u_batch) in train_dataloader:
            optimizer.zero_grad()
            # Forward pass through network
            pred = model(u_batch, xt_batch)
            loss = loss_fn(pred, y_batch.view(-1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        epoch_train_losses.append(np.mean(train_losses))

        # Validation
        model.eval()
        if epoch % 10 == 0:
            validation_losses = []
            model.eval()
            for (xt_batch, y_batch, u_batch) in validation_dataloader:
                pred = model(u_batch, xt_batch)
                loss = loss_fn(pred, y_batch.view(-1))
                validation_losses.append(loss.item())

        
            epoch_val_losses.append(np.mean(validation_losses))
        
        print(f'Epoch {epoch+1}/{epochs}, train loss: {epoch_train_losses[-1]}, val loss: {epoch_val_losses[-1]}')


    return model, epoch_train_losses, epoch_val_losses


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='data/default')
    args.add_argument('--model', type=str, default='FFNN')
    args.add_argument('--n_hidden', type=int, default=128)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=100)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--n_points', type=int, default=100)
    args.add_argument('--outputfolder', type=str, default='default')
    args = args.parse_args()

    root = os.getcwd()
    # print(f'HEEEEEEEY: {root + args.dataset}')

    model, train_losses, val_losses = train_model(args)

    output_folder = args.outputfolder
    
    print(f'Saving model and results to {output_folder}')

    save_model(model, root, output_folder)
    save_results(train_losses, root, output_folder, 'train_losses')
    save_results(val_losses, root, output_folder, 'val_losses')




