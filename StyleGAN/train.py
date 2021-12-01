import argparse
import numpy as np
import torch
from torch.distributions import Normal
import torch.optim as optim
from module.flow import cnf
from datetime import datetime
from torch import nn
from sklearn.utils import shuffle
from NICE import NiceFlow

from utils import load_dataset, make_dir, iterate_batches, load_model, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--values", type=str, default="continuous")
parser.add_argument("--sigma", type=float, default=0.4)
parser.add_argument("--decay", type=float, default=0.999)
parser.add_argument("--age-sigma", type=float, default=0.3)
parser.add_argument("--start-epoch", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nice", action='store_true')
args = parser.parse_args()
args.nice = True
print(args)


torch.random.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 1000 if args.nice else 15
batch_size = 32 if args.nice else 5
lr = args.lr
balanced = True if args.values != "continuous" else False
features = 8

decay = args.decay
sigma = args.sigma
age_sigma = args.age_sigma
if args.values == "2": 
    values = 2
    all_w, all_a = load_dataset(values=values)
elif args.values == "continuous":
    values = "continuous"
    all_w, all_a = load_dataset(values=args.values)
else:
    values = [int(args.values[i]) for i in range(len(args.values))]
    all_w, all_a = load_dataset(values=[0] * 9 + values)

model_dir = f"feat{features}_values{args.values}_decay{decay}_sigma{sigma}"

model_dir = f"saved/{model_dir}"

make_dir(model_dir)

zero_padding = torch.zeros(1, 1, 1).to(device)
if args.nice:
    model = NiceFlow(input_dim=512, n_layers=args.layers, n_couplings=4, hidden_dim=512)
else:
    layers = "-".join(["512"] * args.layers)
    model = cnf(512, layers, 1, 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print(model)
# Loading
start_epoch = args.start_epoch
if start_epoch > 0:
    model, optimizer = load_model(
        f"{model_dir}/model_e{start_epoch}.pch", model, optimizer
    )

all_w, all_a = all_w[500:], all_a[500:]
print(all_w.shape, all_a.shape, values)
if values == 2:
    var_ratio = torch.sum((all_a[:, 9:17] > 0).float(), dim=0) / float(all_a.shape[0])
elif values != "continuous":
    var_ratio = []
    for i in range(len(values)):
        curr_var_ratio = torch.zeros(values[i])
        for v in range(values[i]):
            curr_var_ratio[v] = (
                torch.sum(
                    (
                        abs(all_a[:, 9 + i] - (-1 + v * (2.0 / (values[i] - 1)))) < 1e-3
                    ).float(),
                    dim=0,
                )
                / float(all_a.shape[0])
            )
        var_ratio += [curr_var_ratio]

N_dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

for epoch in range(start_epoch, num_epochs):
    all_w, all_a = shuffle(all_w, all_a)
    model.train()
    running_loss, count_batches, loglike = 0.0, 0, 0.0
    for w, a in iterate_batches(all_w, all_a, batch_size):
        w, a = w.to(device)[:, 0, 0:1, :], a.to(device)[:, 9 : 9 + features]
        cond = torch.zeros(a.shape[0], 1, 1, 1, device=device)

        # Distribution
        current_sigma = sigma * (decay ** epoch) * torch.ones_like(a, device=device)
        if balanced:
            if values == 2:
                sigmas_pos = current_sigma * torch.sqrt(2 * var_ratio).to(device)
                sigmas_neg = current_sigma * torch.sqrt(2 * (1 - var_ratio)).to(device)
                current_sigma = torch.where(a > 0, sigmas_pos, sigmas_neg)
            else:
                for i in range(len(values)):
                    for v in range(values[i]):
                        sigma_v = current_sigma[:, i] * torch.sqrt(
                            sum(var_ratio[i]>0) * var_ratio[i][v]
                        ).to(device)
                        current_sigma[:, i] = torch.where(
                            abs(a[:, i] - (-1 + v * (2.0 / (values[i] - 1)))) < 1e-3,
                            sigma_v,
                            current_sigma[:, i],
                        )
        if balanced and values == 2: current_sigma[:,6] = age_sigma*(decay**epoch)
        current_mean = a.unsqueeze(1).squeeze(-1)
        current_sigma = current_sigma.unsqueeze(1).squeeze(-1)
        a_dist = Normal(current_mean, current_sigma)

        optimizer.zero_grad()
        if args.nice:
            z, logdet = model(w)
        else:
            z, delta_logp = model(w, cond, zero_padding)
        z_attr = z[:, :, :features]
        z_rest = z[:, :, features:]

        current_mean = a.unsqueeze(1).squeeze(-1)
        current_sigma = current_sigma.unsqueeze(1).squeeze(-1)
        a_dist = Normal(current_mean, current_sigma)
        logp_attr = a_dist.log_prob(z_attr)
        logp_rest = N_dist.log_prob(z_rest)

        logpz = logp_attr.sum(-1, keepdim=True) + logp_rest.sum(-1, keepdim=True)
        if args.nice:
            loss = (logpz + logdet).mean()
        else:
            loss = (logpz - delta_logp).mean()
        (-loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        loglike = torch.mean(logpz).item()
        count_batches += 1
        if (count_batches + 1) % 100 == 0:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(
                f"{dt_string} | Epoch {epoch+1} | Batch {count_batches+1} | Loss {running_loss/count_batches:.4f} | Loglike {loglike/count_batches:.4f}"
            )
    mod = 50 if args.nice else 1
    if (epoch+1) % mod == 0: save_model(f"{model_dir}/model_e{epoch+1}.pch", model, optimizer)
