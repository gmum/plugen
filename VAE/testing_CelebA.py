import argparse
from datetime import datetime
import operator
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torchvision.utils as vutils
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import io
import torch
import random
import os
import sys
import scipy
from scipy import stats
# from scipy import linalg, matrix
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image,ImageFont,ImageDraw

from Dataset_CelebA import CelebA, index_to_label
from M_ModelAE_Cnn import CnnVae, FlowVae
from M_ModelGan_PatchGan import PatchGan as Gan

parser = argparse.ArgumentParser(description='C0AE for CelebA')
parser.add_argument('-bz', '--batch-size', type=int, default=70,
                    help='input batch size for training (default: 128)')
parser.add_argument('-iz', '--image-size', type=int, default=256,
                    help='size to resize for CelebA pics (default: 256)')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train (default: 600)')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', type=str,
                    help='path to model to load')
parser.add_argument('-nf', type=int, default=64,
                    help='output channel number of the first cnn layer (default: 64)')
parser.add_argument('-ep', type=int, default=1,
                    help='starting ep index for outputs')
parser.add_argument('-fd', '--flow-det', type=str, default="const",
                    choices=['const', 'variable'],
                    help='const or variable determinant')
parser.add_argument('-ls', '--latent-sampling', action="store_true",
                        help='sample in the latent space')
parser.add_argument('--flow-n-layers', type=int, default=4)
parser.add_argument('--flow-n-couplings', type=int, default=4)
parser.add_argument('--flow-hidden-dim', type=int, default=256)
parser.add_argument('-mt', '--model-type', type=str, default="Flow",
                    choices=['Cnn', 'Flow'],
                    help='name of the model for logging purposes')
args = parser.parse_args()

location = "./"

print(args)

celeba_zip = "CelebA_Dataset/img_align_celeba.zip"
celeba_txt = "CelebA_Dataset/list_attr_celeba.txt"
model_save = 'model_save'
output_dir = f'Loaded_Outputs/{datetime.now()}'
os.makedirs(output_dir, exist_ok=True)

print("CelebA zip file: ", os.path.abspath(celeba_zip))
print("CelebA txt file: ", os.path.abspath(celeba_txt))
print("model save: ", os.path.abspath(model_save))
print("output dir: ", os.path.abspath(output_dir))


batch_size = args.batch_size
image_size = args.image_size
nf = args.nf
lr = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = CelebA(celeba_zip, celeba_txt, image_size)
label_size = dataset.label_size

test_data = Subset(dataset, range(182638, 202599))
dataloader_test = DataLoader(test_data, batch_size=120)

flow_kwargs = {
    'n_layers': args.flow_n_layers,
    'n_couplings': args.flow_n_couplings,
    'hidden_dim': args.flow_hidden_dim,
    'det_type': args.flow_det
}

# TODO: support CnnVae
# model = FlowVae(image_size, label_size, nf, nc=3).to(device)
if args.model_type == 'Cnn':
    model = CnnVae(image_size, label_size, nf, nc=3).to(device)
elif args.model_type == 'Flow':
    model = FlowVae(
            image_size, label_size, nf, nc=3,
            reparameterize=args.latent_sampling,
            flow_kwargs=flow_kwargs).to(device)

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
        [(16, -1.7, 1.7), (22, -1.7, 1.7), (30, -1.7, 1.7), (24, 1.7, -1.7), "Beard"],
    ]
    model.eval()
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
    ), f'{location}/{output_dir}/MSP_recon_CelebA0_{epoch}.png', nrow=b_size, padding=4)
    pass

def sample_flip(epoch, class_sigma=1e-8, style_sigma=0.5):
    test_label = [
        # [(8, 2, -2), (9, -2, 2), "Hair_Colour"],
        [(15, -5, 5), "Glasses"],
        # [(4, -2, 2), (5, 2, -2), (28, -2, 2), "Bald_Bangs"],
        [(23, 5, -5), "Narrow_Eyes"],
        [(21, -5, 5), "Mouth_Open"],
        [(31, -5, 5), "Smiling"],
        [(20, -5, 5), "Male"],
        # [(16, -1.7, 1.7), (22, -1.7, 1.7), (30, -1.7, 1.7), (24, 1.7, -1.7), "Beard"],
    ]
    model.eval()
    _, label, _ = iter(dataloader_test).next()
    label = label[0]
    label[label == -1] = 0.
    label = label.bool()

    b_size = 10
    blank = torch.zeros(b_size, 3, image_size, image_size, device=device)

    normal_samples = torch.normal(0, style_sigma, size=(b_size, model.hidden_size)).to(device)
    first_mix_samples = torch.normal(1., class_sigma, size=(b_size, model.label_size))
    second_mix_samples = torch.normal(-1., class_sigma, size=(b_size, model.label_size))
    comp_choice = torch.randint(0, 2, size=(100, 1)).bool()
    # mix_samples = torch.where(comp_choice, first_mix_samples, second_mix_samples)
    mix_samples = torch.where(label, first_mix_samples, second_mix_samples)
    normal_samples[:, :model.label_size] = mix_samples
    inversed_flow = model.flow.inv_flow(normal_samples)
    decoded = model.decoder(inversed_flow)
    decoded = (decoded + 1) / 2


    with torch.no_grad():
        pic = [decoded, blank]
        for tl in test_label:
            for item in make_continue(tl[: -1], 10):
                item = item[0]
                modified_samples = normal_samples.clone()
                modified_samples[:, item[0]] = item[1]
                inversed_flow = model.flow.inv_flow(modified_samples)
                decoded = model.decoder(inversed_flow)
                decoded = (decoded + 1) / 2
                pic.append(decoded)

            img = Image.new("RGB", (image_size, image_size))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 32)
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), tl[1], (255, 255, 255), font=font)

            annotation = torch.tensor(np.array(img)).permute(2, 0, 1)
            annotated_blank = blank.clone()
            annotated_blank[0] = annotation
            pic.append(annotated_blank)

        comparison = torch.cat(pic)

    vutils.save_image(comparison.cpu(
    ), f'{location}/{output_dir}/MSP_flipclass_{epoch}_s{style_sigma}_c{class_sigma}.jpg', nrow=b_size, padding=4)
    pass

def dataset_interpolation(epoch, s_only=False):
    model.eval()
    data, label, _ = iter(dataloader_test).next()
    label = label[0]
    label[label == -1] = 0.
    label = label.bool()

    b_size = 10
    blank = torch.zeros(b_size + 2, 3, image_size, image_size, device=device)

    rec, z, _, _, _ = model.forward(data.cuda())

    with torch.no_grad():
        pic = []
        for idx in range(b_size - 1):
            interp_from = z[idx]
            interp_to = z[idx + 1]

            to_decode = []
            for alpha in np.linspace(0, 1, num=10):
                if s_only:
                    l_part = interp_from[:40]
                    s_part = interp_from[40:] + alpha * (interp_to[40:] - interp_from[40:])
                    to_decode += [torch.cat([l_part, s_part], 0)]  # TODO: check that
                else:
                    to_decode += [interp_from + alpha * (interp_to - interp_from)]
            to_decode = torch.stack(to_decode).cuda()
            inversed_flow = model.flow.inv_flow(to_decode)
            decoded = model.decoder(inversed_flow)
            decoded = (decoded + 1) / 2

            pic.append((data[idx:idx+1].cpu()) + 1 / 2.)
            pic.append((rec[idx:idx+1].cpu()) + 1 / 2.)
            pic.append(decoded.cpu())
            pic.append(blank.cpu())
        comparison = torch.cat(pic)
        s_only_prefix = "_sonly_" if s_only else ""
    vutils.save_image(comparison.cpu(
    ), f'{location}/{output_dir}/MSP_dataset_interp_{s_only_prefix}.png', nrow=b_size + 2, padding=4)
    pass

def sample(epoch, class_sigma=0.1, style_sigma=1):
    from Dataset_CelebA import index_to_label
    model.eval()
    _, label, _ = iter(dataloader_test).next()
    label = label[0]
    label[label == -1] = 0.
    label = label.bool()

    # Force man
    # label[20] = True

    # Force smiling
    # label[31] = True

    # Force blond
    # label[8] = False
    # label[9] = True
    # label[11] = False

    print(label)
    print("Printing attributes!")
    for label_idx, l in enumerate(label):
        print(f"{index_to_label[label_idx]}: {l}")

    with torch.no_grad():
        normal_samples = torch.normal(0, style_sigma, size=(100, model.hidden_size)).to(device)
        if isinstance(model, FlowVae):
            first_mix_samples = torch.normal(1., class_sigma, size=(100, model.label_size))
            second_mix_samples = torch.normal(-1., class_sigma, size=(100, model.label_size))
            comp_choice = torch.randint(0, 2, size=(100, 1)).bool()
            # mix_samples = torch.where(comp_choice, first_mix_samples, second_mix_samples)
            mix_samples = torch.where(label, first_mix_samples, second_mix_samples)

            normal_samples[:, :model.label_size] = mix_samples
            latent = model.flow.inv_flow(normal_samples)
        elif isinstance(model, CnnVae):
            # zM = normal_samples @ model.M.t()
            # zM[:, :40] = label
            # latent = zM @ model.M
            latent = normal_samples

        decoded = model.decoder(latent)
        decoded = (decoded + 1) / 2

    print("Max and min", decoded.max(), decoded.min())
    vutils.save_image(decoded.cpu(
    ), f'{output_dir}/MSP_CelebA_sampling_test_cs{class_sigma}_ss{style_sigma}.jpg', nrow=10, padding=4)

def sample_flipclass(epoch, class_sigma=0.1, style_sigma=1):
    from Dataset_CelebA import index_to_label
    model.eval()
    _, label, _ = iter(dataloader_test).next()
    label = label[0]
    label[label == -1] = 0.
    label = label.bool()

    label = label.view(1, -1).repeat(41, 1)
    for att in range(0, 40):
        label[att + 1][att] = (1 - label[att + 1][att])

    print(label)
    print("Printing attributes!")
    for label_idx, l in enumerate(label):
        print(f"{index_to_label[label_idx]}: {l}")

    normal_samples = torch.normal(0, style_sigma, size=(1, model.hidden_size)).to(device).repeat(41, 1)

    first_mix_samples = torch.normal(1., class_sigma, size=(41, model.label_size))
    second_mix_samples = torch.normal(-1., class_sigma, size=(41, model.label_size))
    comp_choice = torch.randint(0, 2, size=(41, 1)).bool()
    # mix_samples = torch.where(comp_choice, first_mix_samples, second_mix_samples)
    mix_samples = torch.where(label, first_mix_samples, second_mix_samples)

    normal_samples[:, :model.label_size] = mix_samples
    inversed_flow = model.flow.inv_flow(normal_samples)
    decoded = model.decoder(inversed_flow)
    decoded = (decoded + 1) / 2

    print("Max and min", decoded.max(), decoded.min())
    vutils.save_image(decoded.cpu(
    ), f'{output_dir}/MSP_CelebA_sampling_test_cs{class_sigma}_ss{style_sigma}.jpg', nrow=10, padding=4)

def imageView():
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    pics = []
    fid = 0
    font = ImageFont.load_default()
    tt2 = transforms.ToTensor()
    for i, (data, label, _) in enumerate(bar_data):
        with torch.no_grad():
            if i % 6 == 0:
                im = Image.new(mode = "RGB", size = (256, 256*3))
                draw = ImageDraw.Draw(im)
                draw.text((200, 384),f"{i}",(255,255,255),font=font)
                imv = tt2(im)
                pics.append(imv.unsqueeze(0))
            data = data.unsqueeze(0)
            prod1 = model.predict(data, [(15, label[15]*-2)])
            prod2 = model.predict(data, [(20, -2), (16, 1.5), (22, 1.5), (30, 1.5), (24, -1.5)])
            img = torch.cat([data,prod1,prod2], dim=2)
            img.add_(1.0).div_(2.0)
            pics.append(img)
        if len(pics) == 140:
            vutils.save_image(torch.cat(pics), f'{output_dir}/ImageView/ImageView_{fid}.png', nrow=7, padding=4)
            pics = []
            fid = i + 1
    vutils.save_image(pics, f'{output_dir}/ImageView/ImageView_{fid}.png', nrow=7, padding=4)

def imageView2():
    name = "RemoveGlass"
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    pic = []
    ids = []
    fid = 0
    id_record = ""
    for i, (data, label, _) in enumerate(bar_data):
        if label[15] != 1:
            continue
        with torch.no_grad():
            data = data.unsqueeze(0)
            prod1 = model.predict(data, [(15, 0)])
            prod2 = model.predict(data, [(15, -1)])
            prod3 = model.predict(data, [(15, -2)])
            img = torch.cat([data,prod1,prod2,prod3], dim=2)
            img.add_(1.0).div_(2.0)
            pic.append(img)
            ids.append(i)
        if len(pic) == 40:
            vutils.save_image(torch.cat(pic), f'{output_dir}/{name}/Pic_{fid}.png', nrow=10, padding=4)
            id_record += f"Pic_{fid}:\n {str(ids)}\n\n"
            pic = []
            ids = []
            fid+=1
    vutils.save_image(torch.cat(pic), f'{output_dir}/{name}/Pic_{fid}.png', nrow=10, padding=4)
    id_record += f"Pic_{fid}:\n {str(ids)}\n"
    with open(f'{output_dir}/{name}/Ids.txt', "w") as f:
        f.write(id_record)

def imageSelect():
    name = "M_FemaleBeard"
    font = ImageFont.load_default()
    tt2 = transforms.ToTensor()
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    S = []
    pics = []
    with torch.no_grad():
        for i, (data, label, _) in enumerate(bar_data):
            if label[20]<0:
                continue
            data = data.unsqueeze(0)
            prod = model.predict(data, [(20, -2), (16, 1), (22, 1), (30, 1), (24, -1)])
            label_prod, _ = model.encode(prod)
            label_prod = label_prod[0, [20,16,22,30,24]].clamp(-1.,1.)
            score1 = label_prod * torch.tensor([-1.0,1.0,1.0,1.0,-1.0])
            score2 = (data-prod)**2
            score = score1.mean() * (1-score2.mean())
            S.append((i, score))
        S.sort(key = lambda x:x[1], reverse=True)

        bar_data = tqdm(S, total=len(S))
        fid = 1
        for i,_, _ in bar_data:
            im = Image.new(mode = "RGB", size = (256, 256))
            draw = ImageDraw.Draw(im)
            draw.text((200, 200),f"{i}",(255,255,255),font=font)
            imv = tt2(im)
            pics.append(imv.unsqueeze(0))
            data = test_data[i][0].unsqueeze(0)
            prod = model.predict(data, [(20, -2), (16, 1), (22, 1), (30, 1), (24, -1)])
            data.add_(1.0).div_(2.0)
            prod.add_(1.0).div_(2.0)
            pics.append(data)
            pics.append(prod)
            if len(pics) == 360:
                vutils.save_image(torch.cat(pics), f'{output_dir}/{name}/{fid}.png', nrow=9, padding=4)
                pics = []
                fid += 1
        vutils.save_image(torch.cat(pics), f'{output_dir}/{name}/{fid}.png', nrow=9, padding=4)
    pass



def randnGenerate():
    for i, (data, label, _) in enumerate(dataloader_test):
        label_size = label.shape[1]
        z, _ = model.encode(data.to(device))
        break
    l = z[20:60,:label_size]
    s = torch.randn([40,2048-40]).to(device)
    CONST = 20
    z7 = torch.cat([l,s*0.018 * CONST], dim=1)
    z6 = torch.cat([l,s*0.016 * CONST], dim=1)
    z0 = torch.cat([l,s*0.014 * CONST], dim=1)
    z1 = torch.cat([l,s*0.012 * CONST], dim=1)
    z2 = torch.cat([l,s*0.01 * CONST], dim=1)
    z3 = torch.cat([l,s*0.008 * CONST], dim=1)
    z4 = torch.cat([l,s*0.006 * CONST], dim=1)
    z5 = torch.cat([l,s*0.004 * CONST], dim=1)

    if isinstance(model, FlowVae):
        z0 = model.flow.inv_flow(z0)
        z1 = model.flow.inv_flow(z1)
        z2 = model.flow.inv_flow(z2)
        z3 = model.flow.inv_flow(z3)
        z4 = model.flow.inv_flow(z4)
        z5 = model.flow.inv_flow(z5)
        z6 = model.flow.inv_flow(z6)
        z7 = model.flow.inv_flow(z7)

    prod0 = model.decoder(z0).to("cpu")
    prod1 = model.decoder(z1).to("cpu")
    prod2 = model.decoder(z2).to("cpu")
    prod3 = model.decoder(z3).to("cpu")
    prod4 = model.decoder(z4).to("cpu")
    prod5 = model.decoder(z5).to("cpu")
    prod6 = model.decoder(z6).to("cpu")
    prod7 = model.decoder(z7).to("cpu")
    pics = torch.cat([prod7,prod6,prod0,prod1,prod2,prod3,prod4,prod5])
    pics.add_(1.0).div_(2.0)
    vutils.save_image(pics, f'{output_dir}/randnGenerate.jpg', nrow=40, padding=4)

def jianbian():
    for i, (data, label, _) in enumerate(dataloader_test):
        z, _ = model.encode(data)
        break
    z0 = z[0:20,:]
    z5 = z[20:40,:]
    z1 = z5*0.2 + z0*0.8
    z2 = z5*0.4 + z0*0.6
    z3 = z5*0.6 + z0*0.4
    z4 = z5*0.8 + z0*0.2
    prod0 = model.decoder(z0)
    prod1 = model.decoder(z1)
    prod2 = model.decoder(z2)
    prod3 = model.decoder(z3)
    prod4 = model.decoder(z4)
    prod5 = model.decoder(z5)
    pics = torch.cat([prod0,prod1,prod2,prod3,prod4,prod5])
    pics.add_(1.0).div_(2.0)
    vutils.save_image(pics, f'{output_dir}/Jianbian.jpg', nrow=20, padding=4)

def findImage():
    data_set = test_data
    def load_img(path):
        tt1 = transforms.Resize((225,225))
        tt2 = transforms.ToTensor()
        tt3 = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        with open(path,"rb") as f:
            img = Image.open(f)
            img = tt1(img)
            img = tt2(img)
            img = img[:3]
            img = tt3(img)
        return img
    imgs = [f"{i+1}.png" for i in range(4)]
    imgs = [load_img(i) for i in imgs]

    rst = []
    total_it = len(data_set)
    bar_data = tqdm(data_set, total=total_it)

    for i, p, _ in enumerate(bar_data):
        sc = min([((p[0]-i)**2).sum() for i in imgs])
        rst.append((i,sc))
    rst.sort(key = lambda x:x[1])
    rst = [i for i,_ in rst[:60]]

    print(rst)

    batch = [data_set[i][0] for i in rst]
    batch = torch.stack(batch)
    batch.add_(1.0).div_(2.0)
    vutils.save_image(batch, f'{output_dir}/FindImage.jpg', nrow=10, padding=4)

    pass

def label_morphing(step=5):
    test_label = [
        [(8, 2, -2), (9, -2, 2), "Hair_Colour"],
        [(15, -2, 2), "Glasses"],
        # [(3, 2, -2), "Bags_Under_Eyes"],
        # [(26, -2, 2), "Pale_Skin"],
        # [(4, -2, 2), (5, 2, -2), (28, -2, 2), "Bald_Bangs"],
        # [(23, 2, -2), "Narrow_Eyes"],
        # [(36, 2, -2), "Lipstick"],
        # [(21, -2, 2), "Mouth_Open"],
        # [(31, -2, 2), "Smiling"],
        [(20, -2, 2), "Male"],
        # [(16, -1.7, 1.7), (22, -1.7, 1.7), (30, -1.7, 1.7), (24, 1.7, -1.7), "Beard"],
        # [(16, -2, 2), (22, -2, 2), (30, -2, 2), (24, 2, -2), "Beard"],
        # [(16, 0, 2), (22, 0, 2), (30, 0, 2), (24, 2, -2), (15, 0, 2), "Glasses_Beard"],
    ]
    # [0] := 182638
    test_data = Subset(dataset, range(182950, 182950+6))
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    imgs = []
    for i, (data, label, _) in enumerate(bar_data):
        # imgs.append(data.add(1.0).div(2.0))
        with torch.no_grad():
            data = data.unsqueeze(0)
            for m in test_label:
                for s in range(step):
                    new_label1 = [(i,1.3*(a*(step-1-s)+b*s)/(step-1)) for i,a,b in m[:-1]]
                    prod = model.predict(data, new_label1).squeeze()
                    prod.add_(1.0).div_(2.0)
                    imgs.append(prod)
            vutils.save_image(data.add(1.0).div(2.0), f'{output_dir}/Morphing_Labels/Morphing_label_orgi_{i}.jpg',padding=0)
    img = torch.stack(imgs)
    vutils.save_image(img, f'{output_dir}/Morphing_Labels/Morphing_label.jpg', nrow=step, padding=4)

# TODO: puscic to koniecznie
def pic_morphing(step=7):
    imgs = []
    ids = [(182950,182960), (182952,182962), (182953,182963), (182980,182990), (182950+57,182950+77), (182950+59,182950+79), (182950+60,182950+80), (182950+63,182950+83),
            (182950+100,182950+120),(182950+101,182950+121),(182950+103,182950+123),(182950+109,182950+129),(182950+110,182950+130),(182950+111,182950+131)]
    for i,(i1,i2) in enumerate(ids):
        with torch.no_grad():
            data1 = dataset[i1][0].unsqueeze(0)
            data2 = dataset[i2][0].unsqueeze(0)
            z1,_ = model.encode(data1)
            z2,_ = model.encode(data2)
            for s in range(step):
                z = (z1*(step-1-s)+z2*s)/(step-1)
                prod = model.decoder(z)
                prod.add_(1.0).div_(2.0)
                imgs.append(prod.squeeze())
            vutils.save_image(data1.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/Morphing_pic1_orgi_{i}.jpg',padding=0)
            vutils.save_image(data2.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/Morphing_pic2_orgi_{i}.jpg',padding=0)
    img = torch.stack(imgs)
    vutils.save_image(img, f'{output_dir}/Morphing_Pics/Morphing_Pics.jpg', nrow=step, padding=4)

def features_histograms(ep=0):
    model.eval()
    with torch.no_grad():
        total_it = len(dataloader_test)
        bar_data = tqdm(dataloader_test, total=total_it)
        X = np.empty(shape=(19962, 80))
        idx = 0
        for (data, labels, _) in bar_data:
            data = data.to(device)
            X[idx:idx+data.shape[0],0:40] = labels.numpy()
            latent, _ = model.encode(data)
            latent, _ = model.flow(latent)
            X[idx:idx+data.shape[0],40:] = latent[:,:40].cpu().detach().numpy()
            idx += data.shape[0]

        plt.rcParams.update({'font.size': 31})
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = np.ravel(axes)
        # for plt_idx, class_idx in enumerate([15, 35, 39, 23]):
        for plt_idx, class_idx in enumerate([15, 35, 36, 39, 2, 23]):
            pos = X[X[:,class_idx]==1][:,40+class_idx]
            neg = X[X[:,class_idx]==-1][:,40+class_idx]

            kde_pos = stats.gaussian_kde(pos)
            kde_neg = stats.gaussian_kde(neg)

            # axes[plt_idx].hist(neg, bins=80, alpha=0.5, label="Negative", color="red", density=True)
            # axes[plt_idx].hist(pos, bins=80, alpha=0.5, label="Positive", color="blue", density=True)

            xx = np.linspace(-2.5, 2.5, num=150)
            axes[plt_idx].plot(xx, kde_neg(xx), linewidth=3., label="Negative", color="C0")
            axes[plt_idx].fill_between(xx, kde_neg(xx), color="C0", alpha=0.5)

            axes[plt_idx].plot(xx, kde_pos(xx), linewidth=3., label="Positive", color="C1")
            axes[plt_idx].fill_between(xx, kde_pos(xx), color="C1", alpha=0.5)

            if plt_idx == 0:
                axes[0].legend(loc='upper left')

            if plt_idx == 0:
                axes[plt_idx].set_ylabel("Binary-like Attributes")
            if plt_idx == 3:
                print("We're here", plt_idx)
                axes[plt_idx].set_ylabel("Continuous-like Attributes")

            if plt_idx % 3:
                axes[plt_idx].get_yaxis().set_visible(False)
            else:
                axes[plt_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            axes[plt_idx].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-2., -1, 0, 1, 2.]))


            axes[plt_idx].set_ylim(0, 1)
            axes[plt_idx].set_xlim([-2.5, 2.5])
            #plt.axvline(x=np.mean(neg), color="red")
            #plt.axvline(x=np.mean(pos),color="blue")
            axes[plt_idx].set_title(f"{index_to_label[class_idx].replace('_', ' ')}")
        plt.tight_layout()

        fig.savefig(f"{output_dir}/histogram_{ep:03}.jpg")
        # vutils.save_image(imgs, f"{output_dir}/histogram_{ep:03}.jpg", nrow=5, padding=2, pad_value=1)

def buffer_plot_and_get():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


print("loading model...")
model.load_state_dict(torch.load(f'{args.load}',map_location='cpu'))
model.eval()

features_histograms()

# # print("processing...")
# for style_sigma in [1., 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 1e-2]:
#     for class_sigma in [0.1, 1e-2]:
#         sample(-1, class_sigma, style_sigma)
# for style_sigma in [1., 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 1e-2]:
#     for class_sigma in [0.1, 1e-2]:
#         sample_flip(-1, class_sigma, style_sigma)
# dataset_interpolation(-1)
# dataset_interpolation(-1, s_only=True)
# randnGenerate()
# 
# reconstruction(-1)
# imageView()
# jianbian()
# randnGenerate()
# findImage()
# imageView2()
# imageSelect()
# label_morphing(7)
# pic_morphing()
# pass
