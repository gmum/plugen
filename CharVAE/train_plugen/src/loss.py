import torch
from torch.nn import functional as F
import torch.distributions as D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_vae(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def loss_msp(label, flow_z):
    label_size = label.size(1)
    y_pred = flow_z[:,:label_size]
    s = flow_z[:,label_size:]

    L1 = F.mse_loss(y_pred, label, reduction="sum")
    L2 = (s ** 2).sum()
    return L1, L2
    
    
def loss_reconstruction(criterion, input, decoded):
    _target = input.transpose(0,1).flatten()
    _decoded = decoded.reshape(-1,decoded.size(2))
    return criterion(_decoded, _target)


def loss_flow_classification(label, z, sigma):
    label_size = label.size(1)
    zl = z[:, :label_size]
    s = z[:, label_size:]

    sigma_tensor = torch.tensor([sigma], device=device)

    # Mixture for class variables
    first_gaussian = D.Normal(
        torch.tensor([1.], device=device), sigma_tensor)
    second_gaussian = D.Normal(
        torch.tensor([-1.], device=device), sigma_tensor)

    first_log_prob = first_gaussian.log_prob(zl)
    second_log_prob = second_gaussian.log_prob(zl)

    bool_label = label.clone()
    bool_label[bool_label == -1] = 0.
    bool_label = bool_label.bool()

    loglike_zl = torch.where(bool_label, first_log_prob, second_log_prob)

    # One Gaussian for the rest
    standard_normal = D.Normal(
        torch.tensor([0.], device=device),
        torch.tensor([1.], device=device))
    loglike_s = standard_normal.log_prob(s)

    loglike = (loglike_s.sum(-1) + loglike_zl.sum(-1)).mean()
    return loglike


def loss_flow_regression(label, z, sigma):
    label_size = label.size(1)
    zl = z[:, :label_size]
    s = z[:, label_size:]
    
    data_gaussian = D.MultivariateNormal(label, sigma * torch.eye(label_size, device=device))
    loglike_zl = data_gaussian.log_prob(zl)
    
    # One Gaussian for the rest
    standard_normal = D.Normal(
        torch.tensor([0.], device=device),
        torch.tensor([1.], device=device))
    loglike_s = standard_normal.log_prob(s)

    loglike = (loglike_s.sum(-1) + loglike_zl.sum(-1)).mean()
    return loglike

    
def msp_vae_loss(criterion, input, decoded, mu, logvar, label, flow_z, task='regression', alpha=1.0, sigma=0.1):
    L_rec = loss_reconstruction(criterion, input, decoded)
    L_vae = loss_vae(mu, logvar)
    L_msp_1, L_msp_2 = loss_msp(label, flow_z)
    if task == 'regression':
        L_flow = loss_flow_regression(label, flow_z, sigma)
    elif task == 'classification':
        L_flow = loss_flow_classification(label, flow_z, sigma)
    else:
        raise Exception('Wrong task selected!')
    
    _msp_weight = input.numel()/(label.numel()+flow_z.numel())
    Loss = L_rec + L_vae + (L_msp_1+L_msp_2) * _msp_weight - alpha * L_flow
    return Loss, L_rec, L_vae, L_msp_1, L_msp_2, L_flow
