import argparse
import math
from tqdm import tqdm
import operator
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import torch
import random
import os
import sys
import scipy
from scipy import linalg, matrix
import time

from Dataset_CelebA import CelebA
from M_ModelAE_Cnn import CnnVae, FlowVae
from M_ModelGan_PatchGan import PatchGan as Gan

parser = argparse.ArgumentParser(description='C0AE for CelebA')
parser.add_argument('-bz', '--batch-size', type=int, default=70,
                    help='input batch size for training (default: 70)')
parser.add_argument('-iz', '--image-size', type=int, default=256,
                    help='size to resize for CelebA pics (default: 256)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--pretrained-flow', action='store_true')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', type=str, default=None,
                    help='path to load model')
parser.add_argument('-nf', type=int, default=64,
                    help='output channel number of the first cnn layer (default: 64)')
parser.add_argument('-lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('-ep', type=int, default=1,
                    help='starting ep index for outputs')
parser.add_argument('-pg', action='store_true',
                    help='show tqdm bar')
parser.add_argument('-t', '--train', action='store_true',
                    help='train the model')
parser.add_argument('--train-mode', type=str,
                    choices=['ae_only', 'vae_only', 'flow_only', 'all', 'vanilla_msp'],
                    help='choose parts of the model to train')
parser.add_argument('-mt', '--model-type', type=str, default="Cnn",
                    choices=['Cnn', 'Flow'],
                    help='name of the model for logging purposes')
parser.add_argument('-mn', '--model_name', type=str, default="default",
                    help='name of the model for logging purposes')

parser.add_argument('--labels-per-image', type=int, default=5,
                    help='how many labels per image (semi-supervised setting)')
parser.add_argument('--fully-supervised', action="store_true",
                    help='use all the labels (overrides --labels-per-image)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='weight of the hsic loss')
parser.add_argument('--sigma', type=float, default=1.,
                    help='sigma for the hsic loss')
parser.add_argument('--sigma_decay_base', type=float, default=0.99,
                    help='base for sigma decay')

parser.add_argument('-fd', '--flow-det', type=str, default="const",
                    choices=['const', 'variable'],
                    help='const or variable determinant')
parser.add_argument('-ls', '--latent-sampling', action="store_true",
                        help='sample in the latent space')
parser.add_argument('-bp', '--balanced-prior', action="store_true",
                        help='normalize the stds of Gaussians in the prior')
parser.add_argument('--flow-n-layers', type=int, default=4)
parser.add_argument('--flow-n-couplings', type=int, default=4)
parser.add_argument('--flow-hidden-dim', type=int, default=256)
parser.add_argument('--neptune', action="store_true")
args = parser.parse_args()

# args.load = True
# args.save = False
# args.pg = True


print(args)

if args.neptune:
    import neptune
    neptune.init()
    exp = neptune.create_experiment(name=args.model_name, params=vars(args))
    model_name = args.model_name + "_" + exp.id
else:
    model_name = args.model_name

celeba_zip = "CelebA_Dataset/img_align_celeba.zip"
celeba_txt = "CelebA_Dataset/list_attr_celeba.txt"
model_save = args.model_name + '/model_save/'
output_dir = args.model_name + '/Outputs/'

print("CelebA zip file: ", os.path.abspath(celeba_zip))
print("CelebA txt file: ", os.path.abspath(celeba_txt))
print("model save: ", os.path.abspath(model_save))
print("output dir: ", os.path.abspath(output_dir))

if not os.path.isdir(model_save):
    os.makedirs(model_save, exist_ok=True)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


batch_size = args.batch_size
image_size = args.image_size
nf = args.nf
lr = args.lr
# lr = 1e-4
# lr = 1e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE!", device)

dataset = CelebA(
        celeba_zip, celeba_txt, image_size,
        fully_supervised=args.fully_supervised,
        labels_per_image=args.labels_per_image)
label_size = dataset.label_size

train_data = Subset(dataset, range(0, 182638))
test_data = Subset(dataset, range(182638, 202600))

dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_data, batch_size=min(40, batch_size))



flow_kwargs = {
    'n_layers': args.flow_n_layers,
    'n_couplings': args.flow_n_couplings,
    'hidden_dim': args.flow_hidden_dim,
    'det_type': args.flow_det
}

if args.model_type == 'Cnn':
    model = CnnVae(image_size, label_size, nf, nc=3).to(device)
elif args.model_type == 'Flow':
    model = FlowVae(
            image_size, label_size, nf, nc=3,
            reparameterize=args.latent_sampling,
            flow_kwargs=flow_kwargs).to(device)
discriminator = Gan(image_size, nf=nf, layer=int(math.log2(image_size)-5)).to(device)

print(model)
# print(discriminator)
print("Hidden size!", model.hidden_size)

if args.train_mode == 'ae_only':
    model_params = list(model.parameters())
elif args.train_mode == 'flow_only':
    model_params = list(model.flow.parameters())
else:
    model_params = list(model.parameters())

optimizer = optim.Adam(model_params, lr=lr)
optimizer2 = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)

def train(ep):
    model.train()
    discriminator.train()

    epoch_loss = 0
    epoch_loss_rec = 0
    epoch_loss_vae = 0
    epoch_loss_msp_1 = 0
    epoch_loss_msp_2 = 0
    epoch_loss_pch = 0
    epoch_loss_flow = 0
    epoch_acc = 0
    epoch_D0 = []
    epoch_D1 = []
    total = 0

    if args.balanced_prior:
        var_ratio = torch.sum(dataset.labels[:,:40]>0, axis=0)/float(dataset.labels.shape[0])
    else:
        var_ratio = torch.tensor([0.5] * 40)

    bar_data = dataloader_train
    if args.pg:
        total_it = len(dataloader_train)
        bar_data = tqdm(dataloader_train, total=total_it)
    for i, (data, label, mask) in enumerate(bar_data):
        b_size = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        prod, z, mu, logvar, logdet = model(data)

        # train discriminator
        if args.train_mode != "flow_only":
            optimizer2.zero_grad()
            real = discriminator(data)
            fake = discriminator(prod.detach())
            D=(real.mean().item(),fake.mean().item())
            const = torch.ones_like(real)
            loss_real = discriminator.loss(real, const)
            loss_fake = discriminator.loss(fake, const-1)
            loss_gan = loss_real + loss_fake
            loss_gan.sum().backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            optimizer2.step()
            optimizer2.zero_grad()
        else:
            D = (0, 0)
        # end

        alpha = args.alpha

        if args.train_mode != "ae_only" and args.train_mode != "vae_only":
            if args.train_mode == "flow_only" and not args.pretrained_flow:
                progress = (ep - 50) / (args.epochs - 50)
            else:
                progress = ep / args.epochs
            sigma = args.sigma * (args.sigma_decay_base ** progress)
        else:
            sigma = args.sigma

        Loss, l_rec, l_vae, l_msp_1, l_msp_2, l_flow = model.loss(
                prod=prod, orgi=data, label=label, mask=mask, z=z, mu=mu, logvar=logvar,
                var_ratio=var_ratio, logdet=logdet, alpha=alpha, sigma=sigma)
        if args.train_mode != "flow_only":
            fake = discriminator(prod)
            Loss_pch = discriminator.loss(fake, const, False).sum()
        else:
            Loss_pch = torch.zeros([])

        acc = model.acc(z, label)
        epoch_loss += Loss.item()
        epoch_loss_rec += l_rec.item()
        epoch_loss_vae += l_vae.item()
        epoch_loss_msp_1 += l_msp_1.item()
        epoch_loss_msp_2 += l_msp_2.item()
        epoch_loss_pch += Loss_pch.item()
        epoch_loss_flow += l_flow.item()
        epoch_acc += acc
        epoch_D0.append(D[0])
        epoch_D1.append(D[1])
        total += b_size

        if args.pg:
            bar_data.set_description(f"ep{ep} -- Loss: {Loss.item()/b_size:.0f}, loss_rec: {l_rec.item()/b_size:.0f},  loss_vae: {l_vae.item()/b_size:.0f}, loss_flow: {l_flow.item()/b_size:.2f}, loss_msp_1: {l_msp_1.item()/b_size:.2f}, loss_msp_2: {l_msp_2.item()/b_size:.2f}, loss_gan: {Loss_pch.item()/b_size:.0f}|r{D[0]:.3f}|f{D[1]:.3f}, acc: {acc:.4f}")

        if args.train_mode == "vanilla_msp":
            (Loss + Loss_pch).backward()
        elif args.train_mode == "ae_only":
            (l_rec + Loss_pch).backward()
        elif args.train_mode == "vae_only":
            (l_rec + l_vae + Loss_pch).backward()
        elif args.train_mode == "flow_only":
            (-1 * l_flow).backward()
        else:
            (l_rec +  Loss_pch - l_flow).backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1)
        optimizer.step()


    mean_D0,mean_D1 = sum(epoch_D0)/i, sum(epoch_D1)/i
    std_D0,std_D1 = sum([abs(mean_D0-i) for i in epoch_D0])/i, sum([abs(mean_D1-i) for i in epoch_D1])/i
    return epoch_loss/total, epoch_loss_rec/total, epoch_loss_vae/total, epoch_loss_msp_1/total, epoch_loss_msp_2/total, epoch_loss_pch/total, epoch_loss_flow / total, epoch_acc/i, mean_D0,mean_D1,std_D0,std_D1


def make_continue(cl, total=5):
    # e.g.:
    #   cl = [(15,-1,1),(12,2,-2)]
    #   total = 5
    step = [(i[2]-i[1])/(total-1) for i in cl]
    output = []
    for i in range(total):
        o = [(x[0], x[1]+i*step[j]) for j, x in enumerate(cl)]
        output.append(o)
    # output of e.g.:
    # [
    #   [(15, -1), (12, 2)],
    #   [(15, -0.5), (12, 1)],
    #   [(15, 0), (12, 0)],
    #   [(15, 0.5), (12, -1)],
    #   [(15, 1), (12, -2)],
    # ]
    return output


def reconstruction(epoch):
    test_label = [
        [(8, 2, -2), (9, -2, 2), "Hair_Colour"],
        [(15, -2, 2), "Glasses"],
        [(4, -2, 2), (5, 2, -2), (28, -2, 2), "Bald_Bangs"],
        [(23, 2, -2), "Narrow_Eyes"],
        [(21, -2, 2), "Mouth_Open"],
        [(31, -2, 2), "Smiling"],
        [(20, -2, 2), "Male"],
        [(16, -2, 2), (22, -2, 2), (30, -2, 2), (24, 2, -2), "Beard"],
    ]
    # test_label = [
    #     [(0, -2, 2), "Male"],
    #     [(1, -2, 2), "Smiling"],
    #     [(2, -2, 2), "Glasses"],
    # ]
    model.eval()
    discriminator.eval()
    data, _, _ = iter(dataloader_test).next()
    data = data.to(device)
    b_size = data.shape[0]
    blank = torch.zeros(b_size, 3, image_size, image_size, device=device)
    prod = model.predict(data).add_(1.0).div_(2.0)
    with torch.no_grad():
        pic = [(data+1)*0.5, prod, blank]
        for tl in test_label:
            for item in make_continue(tl[: -1], 5):
                prod = model.predict(data, item)
                prod.add_(1.0).div_(2.0)
                pic.append(prod)
            pic.append(blank)
        comparison = torch.cat(pic)

    vutils.save_image(comparison.cpu(
    ), f'{output_dir}/MSP_CelebA_test_{epoch}.jpg', nrow=b_size, padding=4)
    # if isinstance(epoch, int) and (epoch + 1) % 25 == 0:
    #     neptune.log_image('rec', f'{output_dir}/MSP_CelebA_test_{epoch + 1}.jpg')


def sample(epoch, sigma):
    model.eval()
    discriminator.eval()
    with torch.no_grad():

        if isinstance(model, FlowVae):
            normal_samples = torch.randn(100, model.hidden_size).to(device)
            first_mix_samples = torch.normal(-1., sigma, size=(100, model.label_size))
            second_mix_samples = torch.normal(1., sigma, size=(100, model.label_size))
            comp_choice = torch.randint(0, 2, size=(100, 1)).bool()
            mix_samples = torch.where(comp_choice, first_mix_samples, second_mix_samples)

            normal_samples[:, :model.label_size] = mix_samples
            latent = model.flow.inv_flow(normal_samples)
        elif isinstance(model, CnnVae):
            latent = torch.randn(100, model.hidden_size).to(device)


        decoded = model.decoder(latent)
        decoded = (decoded + 1) / 2.

    vutils.save_image(decoded.cpu(
    ), f'{output_dir}/MSP_CelebA_sampling_{epoch}.jpg', nrow=10, padding=4)


if args.load:
    print("loading model...")
    ep_prefix = 32
    # TODO: fix loading
    loaded_state = torch.load(f'{args.load}/MSP_CelebA.tch')

    if args.train_mode == "flow_only":
        print("Loaded", loaded_state.keys(), "Model", model.state_dict().keys())
        if not args.pretrained_flow:
            loaded_state = {k: v for k, v in loaded_state.items() if "flow" not in k}
        missing, unexpected = model.load_state_dict(loaded_state, strict=False)
        print("Missing keys", missing)
        print("Unexpected keys", unexpected)
    else:
        model.load_state_dict(loaded_state)


    if args.train_mode == "flow_only":
        pack = torch.load(f'{args.load}/MSP_CelebA.opt.tch')
        # optimizer.load_state_dict(pack["optimizer"])
        discriminator.load_state_dict(pack["discriminator"])
        # optimizer2.load_state_dict(pack["optimizer2"])
    else:
        pack = torch.load(f'{args.load}/MSP_CelebA.opt.tch')
        optimizer.load_state_dict(pack["optimizer"])
        discriminator.load_state_dict(pack["discriminator"])
        optimizer2.load_state_dict(pack["optimizer2"])
    args.ep = pack["ep"]+1
    model.eval()
    discriminator.eval()
    print("loaded")


if not args.train:
    reconstruction("loaded")
    sample("loaded", args.sigma)
    sys.exit()

reconstruction(0)
sample(0, args.sigma)

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | starting training ...")
for ep in range(args.ep, args.epochs):
    loss, loss_rec, loss_vae, loss_msp_1, loss_msp_2, loss_pch, loss_flow, acc, mean_D0,mean_D1,std_D0,std_D1 = train(ep)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{localtime} | M ep:{ep} == loss: {loss:.0f}, loss_rec: {loss_rec:.0f}, loss_vae: {loss_vae:.0f}, loss_msp_1: {loss_msp_1:.3f}, loss_msp_2: {loss_msp_2:.3f}, loss_pch: {loss_pch:.0f}, acc: {acc:.4f} == Gan: r{mean_D0:.2f}±{std_D0:.2f} | f{mean_D1:.2f}±{std_D1:.2f}")

    if args.neptune:
        neptune.log_metric("loss", loss)
        neptune.log_metric("loss_rec", loss_rec)
        neptune.log_metric("loss_vae", loss_vae)
        neptune.log_metric("loss_pch", loss_pch)
        neptune.log_metric("loss_msp1", loss_msp_1)
        neptune.log_metric("loss_msp2", loss_msp_2)
        neptune.log_metric("loss_flow", loss_flow)
        neptune.log_metric("acc", acc)

    reconstruction(ep)
    sample(ep, args.sigma)
    if args.save:
        torch.save(model.state_dict(), f"{model_save}/MSP_CelebA.tch")
        torch.save({
            "ep" : ep,
            "optimizer" : optimizer.state_dict(),
            "discriminator" : discriminator.state_dict(),
            "optimizer2" : optimizer2.state_dict()
            }, f'{model_save}/MSP_CelebA.opt.tch')
pass
