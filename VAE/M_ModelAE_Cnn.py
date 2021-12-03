import activations
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F
import numpy as np

from M_Flow_NICE import FlowModel

class Encoder(nn.Module):
    # only for square pics with width or height is n^(2x)
    def __init__(self, image_size, nf, hidden_size=None, nc=3):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        sequens = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        while(True):
            image_size = image_size/2
            if image_size > 4:
                sequens.append(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
                sequens.append(nn.BatchNorm2d(nf * 2))
                sequens.append(nn.LeakyReLU(0.2, inplace=True))
                nf = nf * 2
            else:
                if hidden_size is None:
                    self.hidden_size = int(nf)
                sequens.append(nn.Conv2d(nf, self.hidden_size, int(image_size), 1, 0, bias=False))
                break
        self.main = nn.Sequential(*sequens)

    def forward(self, input):
        return self.main(input).squeeze(3).squeeze(2)


class Decoder(nn.Module):
    # only for square pics with width or height is n^(2x)
    def __init__(self, image_size, nf, hidden_size=None, nc=3):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        sequens = [
            nn.Tanh(),
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
        ]
        while(True):
            image_size = image_size/2
            sequens.append(nn.ReLU(True))
            sequens.append(nn.BatchNorm2d(nf))
            if image_size > 4:
                sequens.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False))
            else:
                if hidden_size is None:
                    self.hidden_size = int(nf)
                sequens.append(nn.ConvTranspose2d(self.hidden_size, nf, int(image_size), 1, 0, bias=False))
                break
            nf = nf*2
        sequens.reverse()
        self.main = nn.Sequential(*sequens)

    def forward(self, z):
        z = z.unsqueeze(2).unsqueeze(2)
        output = self.main(z)
        return output

    def loss(self, predict, orig):
        batch_size = predict.shape[0]
        a = predict.view(batch_size, -1)
        b = orig.view(batch_size, -1)
        L = F.mse_loss(a, b, reduction='sum')
        return L


class CnnVae(nn.Module):
    def __init__(self, image_size, label_size, nf, hidden_size=None, nc=3):
        super(CnnVae, self).__init__()
        self.encoder = Encoder(image_size, nf, hidden_size)
        self.decoder = Decoder(image_size, nf, hidden_size)
        self.image_size = image_size
        self.nc = nc
        self.label_size = label_size
        self.hidden_size = self.encoder.hidden_size

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.M = nn.Parameter(torch.empty(label_size, self.hidden_size))
        nn.init.xavier_normal_(self.M)


    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        prod = self.decoder(z)
        return prod, z, mu, logvar, 0

    def _loss_vae(self, mu, logvar):
        # https://arxiv.org/abs/1312.6114
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    # TODO: ignore masks for now
    def _loss_msp(self, label, mask, z):
        y_pred = z @ self.M.t()
        L1 = F.mse_loss(y_pred, label, reduction="none")
        L1 = L1[mask].sum()

        y_combined = torch.where(mask, label, y_pred)
        L3 = F.mse_loss((y_combined @ self.M), z, reduction="none").sum()

        return L1, L3

    def loss(self, prod, orgi, label, mask, z, mu, logvar, var_ratio, logdet, alpha, sigma):
        L_rec = self.decoder.loss(prod, orgi)
        L_vae = self._loss_vae(mu, logvar)
        L_msp = self._loss_msp(label, mask, z)
        _msp_weight = orgi.numel()/(label.numel()+z.numel())
        Loss = L_rec + L_vae + (L_msp[0] + L_msp[1]) * _msp_weight
        L_flow = torch.zeros([])
        return Loss, L_rec, L_vae, L_msp[0], L_msp[1], L_flow

    def acc(self, z, l):
        zl = z @ self.M.t()
        a = zl.clamp(-1, 1)*l*0.5+0.5
        return a.round().mean().item()

    def predict(self, x, new_ls=None, weight=1.0):
        z, _ = self.encode(x)
        if new_ls is not None:
            zl = z @ self.M.t()
            d = torch.zeros_like(zl)
            for i, v in new_ls:
                d[:,i] = v*weight - zl[:,i]
            z += d @ self.M
        prod = self.decoder(z)
        return prod

    def predict_ex(self, x, label, new_ls=None, weight=1.0):
        return self.predict(x,new_ls,weight)

class FlowVae(nn.Module):
    def __init__(self, image_size, label_size, nf, hidden_size=None, nc=3, reparameterize=False, flow_kwargs=None):
        super(FlowVae, self).__init__()
        self.encoder = Encoder(image_size, nf, hidden_size)
        self.decoder = Decoder(image_size, nf, hidden_size)
        self.image_size = image_size
        self.nc = nc
        self.label_size = label_size
        self.hidden_size = self.encoder.hidden_size

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.M = nn.Parameter(torch.empty(label_size, self.hidden_size))
        nn.init.xavier_normal_(self.M)

        self.flow = FlowModel(self.hidden_size, **flow_kwargs)
        self.reparam_required = reparameterize

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.reparam_required:
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        prod = self.decoder(latent)
        z, logdet = self.flow(latent)
        return prod, z, mu, logvar, logdet

    def _loss_vae(self, mu, logvar):
        # https://arxiv.org/abs/1312.6114
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def _loss_msp(self, label, mask, z):
        y_pred = z[:, :self.label_size]
        s = z[:, self.label_size:]

        L1 = F.mse_loss(y_pred, label, reduction="none")
        L1 = L1[mask].sum()

        L2 = (s ** 2).sum()
        return L1, L2

    def _loss_flow(self, label, mask, z, sigma, var_ratio, logdet):
        zl = z[:, :self.label_size]
        s = z[:, self.label_size:]

        sigma_pos = (sigma * torch.sqrt(2 * var_ratio)).cuda()
        sigma_neg = (sigma * torch.sqrt(2 * (1 - var_ratio))).cuda()
        # Mixture for class variables
        first_gaussian = D.Normal(
            torch.tensor([1.], device='cuda'), sigma_pos)
        second_gaussian = D.Normal(
            torch.tensor([-1.], device='cuda'), sigma_neg)

        first_log_prob = first_gaussian.log_prob(zl)
        second_log_prob = second_gaussian.log_prob(zl)

        first_prob = first_log_prob.exp()
        second_prob = second_log_prob.exp()

        bool_label = label.clone()
        bool_label[bool_label == -1] = 0.
        bool_label = bool_label.bool()

        class_loglike = torch.where(bool_label, first_log_prob, second_log_prob)
        nonclass_loglike = torch.log((first_prob + second_prob) / 2 + 1e-6)

        loglike_zl = torch.where(mask, class_loglike, nonclass_loglike)

        # One Gaussian for the rest
        standard_normal = D.Normal(
            torch.tensor([0.], device='cuda'),
            torch.tensor([1.], device='cuda'))
        loglike_s = standard_normal.log_prob(s)

        loglike = (loglike_s.sum(-1) + loglike_zl.sum(-1)) + logdet
        loglike = loglike.mean()
        return loglike


    def loss(self, prod, orgi, label, mask, z, mu, logvar, var_ratio, logdet, alpha=1.0, sigma=0.1):
        L_rec = self.decoder.loss(prod, orgi)
        L_vae = self._loss_vae(mu, logvar)
        L_msp_1, L_msp_2 = self._loss_msp(label, mask, z)
        L_flow = self._loss_flow(label, mask, z, sigma, var_ratio, logdet)

        # TODO: divide by mask numel
        _msp_weight = orgi.numel() / (mask.numel()+z.numel())
        Loss = L_rec + L_vae + (L_msp_1 + L_msp_2) * _msp_weight - alpha * L_flow
        return Loss, L_rec, L_vae, L_msp_1, L_msp_2, L_flow

    def acc(self, z, l):
#         zl = z @ self.M.t()
        zl = z[:,:self.label_size]
        a = zl.clamp(-1, 1)*l*0.5+0.5
        return a.round().mean().item()

    def predict(self, x, new_ls=None, weight=1.0):
        latent, _ = self.encode(x)
        z, _ = self.flow(latent)
#         print(z.shape)

        if new_ls is not None:
#             zl = z @ self.M.t()
            zl = z[:,:self.label_size]
            d = torch.zeros_like(zl)
            for i, v in new_ls:
                d[:,i] = v*weight - zl[:,i]
#             z += d @ self.M
#             print(d.shape)
#             print(d[0])
#             z += d @ self.M
            z[:,:self.label_size] += d

        latent = self.flow.inv_flow(z)
        prod = self.decoder(latent)
        return prod

    def predict_ex(self, x, label, new_ls=None, weight=1.0):
        return self.predict(x,new_ls,weight)

