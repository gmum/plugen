import os

import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import torchvision.utils as vutils

import torch.nn as nn
from tqdm import tqdm

from Dataset_CelebA import CelebA
from sklearn.metrics import f1_score

def train():
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    total = 0

    for i, (data, label, mask) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        label = label.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        label[label == -1] = 0.
        loss_val = loss_fn(outputs, label)  # [batch_size, class_num]
        ratio = torch.where(label.bool(), class_ratio, 1 - class_ratio)
        loss_val = (loss_val / ratio).mean()
        loss_val.backward()
        optimizer.step()

        preds = outputs.sign()
        preds[preds == -1] = 0.
        acc = (preds == label).float().mean()
        epoch_loss += loss_val.item()
        epoch_acc += acc
        total += data.shape[0]

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

def evaluate():
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    total = 0

    all_labels = []
    all_preds = []
    for i, (data, label, mask) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        label[label == -1] = 0.
        loss_val = loss_fn(outputs, label)  # TODO: labels are {-1, 1}
        ratio = torch.where(label.bool(), class_ratio, 1 - class_ratio)
        loss_val = (loss_val * ratio).mean()


        preds = outputs.sign()
        preds[preds == -1] = 0.

        all_labels += [label.cpu().numpy()]
        all_preds += [preds.cpu().detach().numpy()]

        acc = (preds == label).float().mean()
        epoch_loss += loss_val.mean().item()
        epoch_acc += acc
        total += data.shape[0]

    all_labels = np.concatenate(all_labels, 0)
    all_preds = np.concatenate(all_preds, 0)

    print(all_labels.shape, all_preds.shape)

    print("F1 Scores:")
    for class_idx in range(40):
        f1_val = f1_score(all_labels[:, class_idx], all_preds[:, class_idx])
        print(f"Class {class_idx}:\t{f1_val * 100:.2f}")

    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)

parser = argparse.ArgumentParser(description='C0AE for CelebA')
parser.add_argument('-bz', '--batch-size', type=int, default=256,
                    help='input batch size for training (default: 70)')
parser.add_argument('-iz', '--image-size', type=int, default=256,
                    help='size to resize for CelebA pics (default: 256)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--fully-supervised', action="store_true")
parser.add_argument('--lr',type=float, default=3e-4)
parser.add_argument('--load',type=str)
parser.add_argument('--neptune', action="store_true")
args = parser.parse_args()

# args.load = True
# args.save = False
# args.pg = True


print(args)

if args.neptune:
    import neptune
    neptune.init()
    exp = neptune.create_experiment(params=vars(args))

celeba_zip = "CelebA_Dataset/img_align_celeba.zip"
celeba_txt = "CelebA_Dataset/list_attr_celeba.txt"
# model_save = args.model_name + '/model_save/'
# output_dir = args.model_name + '/Outputs/'

print("CelebA zip file: ", os.path.abspath(celeba_zip))
print("CelebA txt file: ", os.path.abspath(celeba_txt))

batch_size = args.batch_size
image_size = args.image_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE!", device)

dataset = CelebA(celeba_zip, celeba_txt, image_size, fully_supervised=args.fully_supervised)
label_size = dataset.label_size

train_data = Subset(dataset, range(0, 182638))
test_data = Subset(dataset, range(182638, 202599))

train_loader = DataLoader(train_data, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=min(40, batch_size))

class_ratio = torch.sum(dataset.labels[:,:40]>0, axis=0)/float(dataset.labels.shape[0])
class_ratio = class_ratio.to(device)

model = models.resnet50(num_classes=40).to(device)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.load:
    model.load_state_dict(torch.load(args.load))
    print(evaluate())
    sys.exit()


for ep in range(args.epochs):
    loss, acc = train()
    with torch.no_grad():
        test_loss, test_acc = evaluate()

    print(f"Train loss {loss}, Train acc {acc}")
    print(f"Test loss {test_loss}, Test acc {test_acc}")
    if args.neptune:
        neptune.log_metric("train_loss", loss)
        neptune.log_metric("train_acc", acc)
        neptune.log_metric("test_loss", test_loss)
        neptune.log_metric("test_acc", test_acc)


    torch.save(model.state_dict(), f"./resnet_classifier_balanced_{image_size}x{image_size}.pt")
