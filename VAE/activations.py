import torch


class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))


class Identity:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x

    def inverse(self, x):
        return x
