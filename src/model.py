import torch

# Neural network 
class DeepONet(torch.nn.Module):
    def __init__(self, n_sensors, n_hidden, n_output):
        super(DeepONet, self).__init__()
        # Branch net
        self.B1 = torch.nn.Linear(n_sensors, n_hidden)
        self.B2 = torch.nn.Linear(n_hidden, n_hidden)
        self.B3 = torch.nn.Linear(n_hidden, n_output)

        # Trunk net
        self.T1 = torch.nn.Linear(2, n_hidden)
        self.T2 = torch.nn.Linear(n_hidden, n_hidden)
        self.T3 = torch.nn.Linear(n_hidden, n_output)

        self.activation = torch.sin

        self.batch = True # Flag for batch training

    def set_batch(self, batch):
        self.batch = batch

    def forward(self, u, y):
        # Pass through branch net
        b = self.activation(self.B1(u))
        b = self.activation(self.B2(b))
        b = self.B3(b)

        # Pass through trunk net
        t = self.activation(self.T1(y))
        t = self.activation(self.T2(t))
        t = self.T3(t)

        # Combine
        if self.batch:
            output = torch.einsum('ij,ij->i', b, t)
        else:
            output = b @ t
            

        return output