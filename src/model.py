import numpy as np
import torch

# Neural network 
class DeepONet(torch.nn.Module):
    def __init__(self, n_sensors, n_output):
        super(DeepONet, self).__init__()
        self.branch = torch.nn.Linear(n_sensors, n_output)
        self.trunk = torch.nn.Linear(2, n_output)

        self.activation = torch.sin

    def forward(self, u, y):

        # Pass IC function through branch net.
        B = self.activation(self.branch(u))

        # Pass point in spacetime through trunk net
        T = self.activation(self.trunk(y))

        # Return real number G(u)(y)
        output = B @ T

        return output