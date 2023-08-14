import torch


class SoftAdapt:
    def __init__(self, num_components, beta=1e-3):
        self.beta = torch.tensor(beta)
        self.epsilon = torch.tensor(1e-8)
        self.history = torch.zeros([num_components, 2])
        self.alphas = torch.ones([num_components])
    
    def get_alphas(self):
        return self.alphas
    
    def update(self, losses):
        self.history[:, 1] = self.history[:, 0]
        self.history[:, 0] = losses

        s = self.history[:, 0] - self.history[:, 1]
        s_max = torch.max(s)
        self.alphas = torch.exp(self.beta * (s - s_max)) / (torch.sum(torch.exp(self.beta * (s - s_max))) + self.epsilon) # ??



if __name__ == "__main__":
    sa = SoftAdapt(2, beta=0.1)
    print(sa.get_alphas())
    sa.update(torch.tensor([1., 20.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([1., 40.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([1., 60.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 1800.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 1800.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 1800.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 1000.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 10.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 10.]))
    print(sa.get_alphas())
    sa.update(torch.tensor([0.1, 10.]))
    print(sa.get_alphas())
