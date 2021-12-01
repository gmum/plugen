import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Coupling layers

class Coupling_layer_NICE(nn.Module):
    def __init__(self, input_dim, n_layers, mask_type, hidden_dim=1024):
        super(Coupling_layer_NICE, self).__init__()

        self.mask = self.get_mask(input_dim, mask_type)

        a = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            a.append(nn.Linear(hidden_dim, hidden_dim))
            a.append(nn.LeakyReLU(0.1))
        a.append(nn.Linear(hidden_dim, input_dim))
        self.a = nn.Sequential(*a)


    def forward(self, x):
        z = x.view(x.shape[0], -1)
        h1, h2 = z * self.mask, z * (1 - self.mask)

        m = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = h2 + m

        z = h1 + h2

        return z.view(x.shape), 0


    def inverse(self, z):
        x = z.view(z.shape[0], -1)
        h1, h2 = x * self.mask, x * (1 - self.mask)

        m = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = h2 - m

        x = h1 + h2

        return x.view(z.shape)


    def get_mask(self, input_dim, mask_type):
        self.mask = torch.zeros(input_dim)
        if mask_type == 0:
            self.mask[::2] = 1

        elif mask_type == 1:
            self.mask[1::2] = 1

        return self.mask.view(1,-1).to(device).float()


class Coupling_layer_det(nn.Module):
    def __init__(self, input_dim, n_layers, mask_type, hidden_dim=1024):
        super(Coupling_layer_det, self).__init__()
        self.mask = self.get_mask(input_dim, mask_type)

        m = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            m.append(nn.Linear(hidden_dim, hidden_dim))
            m.append(nn.LeakyReLU(0.1))
        m.append(nn.Linear(hidden_dim, input_dim))
        m.append(nn.Tanh())
        self.m = nn.Sequential(*m)

        a = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1)]
        for i in range(n_layers-2):
            a.append(nn.Linear(hidden_dim, hidden_dim))
            a.append(nn.LeakyReLU(0.1))
        a.append(nn.Linear(hidden_dim, input_dim))
        self.a = nn.Sequential(*a)

    def forward(self, x):
        z = x.view(x.shape[0], -1)
        h1, h2 = z * self.mask, z * (1 - self.mask)

        m = self.m(h1) * (1 - self.mask)
        a = self.a(h1) * (1 - self.mask)

        #h1 = h1
        h2 = h2 * torch.exp(m) + a

        z = h1 + h2
        logdetJ = torch.sum(m, dim=1)

        return z.view(x.shape), logdetJ


    def inverse(self, z):
        x = z.view(z.shape[0], -1)
        h1, h2 = x * self.mask, x * (1 - self.mask)

        m = self.m(h1) * (1 - self.mask)
        a = self.a(h1) * (1 - self.mask)
        #h1 = h1
        h2 = (h2 - a) * torch.exp(-m)

        x = h1 + h2

        return x.view(z.shape)


    def get_mask(self, input_dim, mask_type):
        self.mask = torch.zeros(input_dim)
        if mask_type == 0:
            self.mask[::2] = 1

        elif mask_type == 1:
            self.mask[1::2] = 1

        return self.mask.view(1,-1).to(device).float()


## Models

class NiceFlow(nn.Module):
    def __init__(self, input_dim, n_layers, n_couplings, hidden_dim, det_type="const"):
        super(NiceFlow, self).__init__()

        self.input_dim = input_dim

        self.coupling_layers = []
        for i in range(n_couplings):
            if det_type == "const":
                layer = Coupling_layer_NICE(input_dim, n_layers, i%2, hidden_dim).float().to(device)
            elif det_type == "variable":
                layer = Coupling_layer_det(input_dim, n_layers, i%2, hidden_dim).float().to(device)

            self.coupling_layers.append(layer)
        self.coupling_layers = nn.ModuleList(self.coupling_layers)


    def forward(self, x):
        return self.flow(x)


    def flow(self, x):
        logdetJ_ac = 0
        x_in = x
        if x_in.shape[1] == 18:
            x = x[:,0,:]
        
        x = x.view(-1, self.input_dim).float()

        for layer in self.coupling_layers:
            x, logdetJ = layer(x)
            logdetJ_ac += logdetJ

        if x_in.shape[1] == 18:
            for i in range(18): x_in[:,i,:] = x
            return x_in, logdetJ
        return x.unsqueeze(1), logdetJ


    def inv_flow(self, z):
        z_in = z
        if z_in.shape[1] == 18:
            z = z_in[:,0,:]
        else: z = z.squeeze(1)
        z = z.view(-1, self.input_dim).float()

        for layer in self.coupling_layers[::-1]:
            z = layer.inverse(z)

        if z_in.shape[1] == 18:
            for i in range(18): z_in[:,i,:] = z
            return z_in
        return z.unsqueeze(1)

    def load_w(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
