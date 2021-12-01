import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import tensorflow as tf
import matplotlib.pyplot as plt
from pretrained_networks import load_networks
from module.flow import cnf
from NICE import NiceFlow
from utils import iterate_batches, load_dataset, make_dir, save_img

parser = argparse.ArgumentParser()
parser.add_argument("--styleflow", action="store_true")
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--values", type=str, default="continuous")
parser.add_argument("--sigma", type=float, default=0.4)
parser.add_argument("--decay", type=float, default=0.999)
parser.add_argument("--age-sigma", type=float, default=0.3)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nice", action="store_true")
args = parser.parse_args()
args.nice = True

styleflow = args.styleflow
num_pics = 50
steps = 10
abso = 2
features = 8
epoch = args.epoch
experiments = [(i, -abso, abso) for i in range(9,17)]
test_set = True

if styleflow:
    prior = cnf(512, "512-512-512-512-512", 8, 1)
    binary = False
    name = "original/feat8_normalized"
    path = f"saved/{name}/model_e{epoch}.pch"
    output_dir = f"outputs/{name}/e{epoch}"
    name = name.replace("/", "_")
    prior.load_state_dict(torch.load(path)["model"])
else:
    if args.nice:
        prior = NiceFlow(input_dim=512, n_layers=args.layers, n_couplings=4, hidden_dim=512)
    else:
        layers = "-".join(["512"] * args.layers)
        prior = cnf(512, layers, 1, 1)
    binary = False
    decay = args.decay
    sigma = args.sigma
    age_sigma = args.age_sigma
    values = args.values
    name = f"feat{features}_values{values}_decay{decay}_sigma{sigma}"
    if args.nice: name = f"nice_{name}"
    prior.load_state_dict(torch.load(f"saved/{name}/model_e{epoch}.pch")["model"])
    output_dir = f"outputs/{name}/e{epoch}"

make_dir(output_dir)
prior.eval()


import dnnlib
import dnnlib.tflib as tflib

network_pkl = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
_, _, Gs = load_networks(network_pkl)
Gs_syn_kwargs = dnnlib.EasyDict()
Gs_syn_kwargs.output_transform = dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
)
Gs_syn_kwargs.randomize_noise = False
Gs_syn_kwargs.minibatch_size = 1

all_w, all_a = load_dataset(keep=False, values=[0] * 17)
#all_w, all_a = all_w[500:], all_a[500:]

def eval_single_change(w, a, change, features, styleflow):
    transformation = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]
    )
    if isinstance(change, tuple):
        attr, val = change
    else: attr = None

    if attr is None:  curr_output_dir = f"{output_dir}/original"
    else: curr_output_dir = f"{output_dir}/{attr:02}_{val}"
    make_dir(curr_output_dir)
    batch_size = min(10, w.shape[0])
    assert w.shape[0] % batch_size == 0
    Gs_syn_kwargs.minibatch_size = batch_size
    rnd = np.random.RandomState(len(w))
    noise_vars = [
        var
        for name, var in Gs.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    zero_padding = torch.zeros(batch_size, 18, 1).to(device)
    for i in range(0, len(w), batch_size):
        print(f"{i}/{len(w)}")
        curr_w, curr_a = w[i : i + batch_size].to(device), a[i : i + batch_size].to(
            device
        )
        new_a = curr_a.clone()
        if attr is not None:
            new_a[:, attr] = val
        cond = (
            torch.zeros(batch_size, 1, 1, 1).to(device)
            if not styleflow
            else curr_a[:, 9:]
        )
        if args.nice:
            z = prior(curr_w)[0]
        else: z = prior(curr_w, cond, zero_padding)[0]
        cond = (
            torch.zeros(batch_size, 1, 1, 1).to(device)
            if not styleflow
            else new_a[:, 9:]
        )
        if not styleflow and (attr is not None):
            for j in range(18):
                z[:, j, 0:features] = new_a[:, 9 : 9 + features, 0]
        if args.nice:
            curr_w = prior.inv_flow(z) 
        else: curr_w = prior(z, cond, zero_padding, True)[0]
        imgs = Gs.components.synthesis.run(
            curr_w.cpu().detach().numpy(), **Gs_syn_kwargs
        )
        for j in range(imgs.shape[0]):
            vutils.save_image(
                transformation(imgs[j]), f"{curr_output_dir}/{i+j:04}.png"
            )


def eval_attribute_change(w, a, change, steps, features, styleflow):
    transform_resize = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((256,256)), transforms.ToTensor()]
    )

    a_source = a.clone()
    a_target = a_source.clone()
    if not isinstance(change, list):
        attr, val_source, val_target = change
        a_source[:, attr] = val_source
        a_target[:, attr] = val_target
    else:
        for attr, val_source, val_target in change:
            a_source[:, attr] = val_source
            a_target[:, attr] = val_target

    rnd = np.random.RandomState(idx)
    noise_vars = [
        var
        for name, var in Gs.components.synthesis.vars.items()
        if name.startswith("noise")
    ]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})

    imgs, ws = [], [w.clone()]
    w_first = w

    img_original = Gs.components.synthesis.run(
        w.cpu().detach().numpy(), **Gs_syn_kwargs
    )[0]
    imgs += [img_original, np.ones_like(img_original)]
    zero_padding = torch.zeros(1, 18, 1).to(device)
    cond = torch.zeros(1, 1, 1, 1).to(device) if not styleflow else a[:, 9:]
    if args.nice:
        z = prior(w)[0]
    else: z = prior(w, cond, zero_padding)[0]
    z_first = z
    for step in range(steps):
        a = torch.lerp(a_source, a_target, step / (steps - 1))
        a = a.squeeze(-1).squeeze(-1)

        if styleflow:
            cond = a[:, 9:]
        else:
            cond = torch.zeros(1, 1, 1, 1).to(device)
            z[:, :, 0:features] = a[:, 9 : 9 + features]
        if args.nice:
            w = prior.inv_flow(z)
        else: w = prior(z, cond, zero_padding, True)[0]
        img = Gs.components.synthesis.run(w.cpu().detach().numpy(), **Gs_syn_kwargs)[0]
        imgs += [img]
        ws += [w.clone()]
        if args.nice:
            z_current = prior(w)[0]
        else: z_current = prior(w, cond, zero_padding)[0]
        w = w_first
        z = z_first
    imgs = [transform_resize(img) for img in imgs]
    return imgs, ws


def plot_histograms(features, values, n=500, batch_size=10):
    all_w, all_a = load_dataset(keep=True, values=[2] * 17, keep_indexes=range(n))
    new_all_a = torch.zeros(len(all_w), features)
    start = 0
    for w, a in iterate_batches(all_w, all_a, batch_size):
        w, a = w[:, 0, 0:1, :].to(device), a[:, 9:].to(device)
        zero_padding = torch.zeros(1, 1, 1, device=device)
        cond = torch.zeros(w.shape[0], 1, 1, 1, device=device)
        if args.nice:
           z = prior(w)[0]
        else: z = prior(w, cond, zero_padding)[0]
        new_all_a[start : start + w.shape[0]] = z[:, 0, 0:features]
        start += z.shape[0]

    titles = [
        "gender",
        "glasses",
        "left-right",
        "up-down",
        "hair",
        "beard",
        "age",
        "expression",
    ]
    for i in range(features):
        bins = 100
        """
        if i != 6:
            plt.hist(
                new_all_a[all_a[:, 9 + i, 0] > 0][:, i].detach().numpy(),
                bins=bins,
                alpha=0.5,
                label="positive",
                color="tab:blue",
                density=True,
            )
            plt.hist(
                new_all_a[all_a[:, 9 + i, 0] < 0][:, i].detach().numpy(),
                bins=bins,
                alpha=0.5,
                label="negative",
                color="tab:red",
                density=True,
            )
        else:
            plt.hist(
                new_all_a[:, i].detach().numpy(), bins=bins, density=True, label="all"
            )
        """
        plt.hist(
                new_all_a[:, i].detach().numpy(),
                bins=bins,
                alpha=0.5,
                label="flow prediction",
                color="tab:blue",
                density=True,
            ) 
        plt.hist(
                all_a[:, 9+i].detach().numpy(),
                bins=bins,
                alpha=0.5,
                label="ground truth",
                color="tab:green",
                density=True,
            )
        plt.title(titles[i])
        plt.legend(loc="upper right")
        plt.savefig(f"{output_dir}/feat_distribution_{i:02}.jpg")
        plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


changes = [ (9,-1),(9,1),(10,-1),(10,1),(13,-1),(13,1),(14,-1),(14,1),(16,-1),(16,1)]
N = 500
for change in changes:
    eval_single_change(all_w[0:N, 0], all_a[0:N], change, features, styleflow)

ws = []
for idx in range(num_pics):
    print(f"Processing img {idx}")
    imgs = []
    for j, change in enumerate(experiments):
        with torch.no_grad():
            w = all_w[idx].to(device)
            a = all_a[idx].to(device).reshape(1, 17, 1, 1)
            current_imgs, current_ws = eval_attribute_change(
                w, a, change, steps=steps, features=features, styleflow=styleflow
            )
            imgs += current_imgs
            ws += [current_ws]
            print(len(imgs))
    if len(imgs) > 0:
        vutils.save_image(
            imgs,
            f"{output_dir}/abs{abso}_{idx}_{name}.png",
            nrow=steps + 2,
            padding=1,
            pad_value=1,
        )


with torch.no_grad():
    if not styleflow:
        plot_histograms(features, values=values)
