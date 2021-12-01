import numpy as np
import pickle
import torch
from PIL import Image
import os
from torch.distributions import Normal
import torch.optim as optim
from module.flow import cnf
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.utils import shuffle

from utils import load_dataset, iterate_batches, save_model, load_model, make_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_w, all_a = load_dataset(values=[0]*9+[1]*8)
all_w, all_a = all_w[500:], all_a[500:]
print(all_w.shape, all_a.shape)

model_dir = "saved/original/feat8_normalized"
model_name = "model"
make_dir(model_dir)

num_epochs = 10
batch_size = 5
lr = 1e-3

# model = cnf(512, "512-512-512-512-512", 17, 1).to(device)
model = cnf(512, "512-512-512-512-512", 8, 1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Loading
start_epoch = 6
if start_epoch > 0:
    model, optimizer = load_model(
        f"{model_dir}/{model_name}_e{start_epoch}.pch", model, optimizer
    )

N_dist = Normal(torch.tensor([0.0], device="cuda"), torch.tensor([1.0], device="cuda"))
zero_padding = torch.zeros(1, 1, 1).to(device)

for epoch in range(start_epoch, num_epochs):
    all_w, all_a = shuffle(all_w, all_a)
    model.train()
    running_loss, count_batches, loglike = 0.0, 0, 0.0
    for w, a in iterate_batches(all_w, all_a, batch_size):
        w, a = w.to(device)[:, 0, 0:1, :], a.to(device)[:, 9:]
        optimizer.zero_grad()
        z, delta_logp = model(w, a, zero_padding)
        
        logpz = N_dist.log_prob(z)
        loss = logpz - delta_logp
        loss = -torch.mean(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loglike += torch.mean(logpz).item()
        count_batches += 1
        if (count_batches + 1) % 100 == 0:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(
                f"{dt_string} | Epoch {epoch+1} | Batch {count_batches+1} | Loss {running_loss/count_batches:.4f} | Loglike {loglike/count_batches:.4f}"
            )

    save_model(f"{model_dir}/{model_name}_e{epoch+1}.pch", model, optimizer)
