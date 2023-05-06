import torch 
from model import DeepONet

"""
    Class used to train and save a DeepONet model:
    Input: 


    Output:
    
"""

def train_model():
    n_sensors = 10 # Number of sensors. Determined from training data 

    model = DeepONet(n_sensors=n_sensors, n_output=1024)
    optimizer = torch.optim.Adam(params=model.parameters, lr=1e-3, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamme=0.96)
    epochs = 10

    return 0

def save_model():
    pass

def save_results():
    pass



if __name__ == '__main__':
    print(train_model())



