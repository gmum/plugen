import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchtext import data

from vae import MSP, VAE, EncoderRNN, DecoderRNN
from flow import FlowModel
from loss import msp_vae_loss
from data_utils import get_dataset
from utils import return_smiles, return_input_smiles


def get_alpha_val(epoch, max_epochs):
    progress = epoch / max_epochs
    if progress < args.alpha_phase1:
        return 1.
    elif progress < args.alpha_phase2:
        duration = args.alpha_phase2 - args.alpha_phase1
        x = (progress - args.alpha_phase1) / duration
        return 1 - x
    else:
        return 0.
    

def run_epoch(data_iter, model, optimizer, alpha, sigma):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    total_l_rec = 0
    total_l_vae = 0
    total_l_msp_1 = 0
    total_l_msp_2 = 0
    total_l_flow = 0

    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        optimizer.zero_grad()
        input = batch.smiles.to(device)
        label = torch.cat((batch.logP.unsqueeze(-1), batch.qed.unsqueeze(-1), batch.SAS.unsqueeze(-1)), axis=1).to(device)
        
        mu, logvar, z, flow_z, decoded = model(input)
        
        loss, l_rec, l_vae, l_msp_1, l_msp_2, l_flow = msp_vae_loss(criterion, input, decoded, mu, logvar, label, flow_z, alpha, sigma)
        
        if args.train_mode == "ae_only":
            (l_rec + l_vae).backward()
        elif args.train_mode == "flow_only":
            (-1 * l_flow).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1)
        optimizer.step()
        
        total_loss += loss.item()
        total_l_rec += l_rec.item()
        total_l_vae += l_vae.item()
        total_l_msp_1 += l_msp_1.item()
        total_l_msp_2 += l_msp_2.item()
        total_l_flow += l_flow.item()
        total_tokens += batch.batch_size
            
    return total_loss / total_tokens, total_l_rec / total_tokens, total_l_vae / total_tokens, total_l_msp_1 / total_tokens, total_l_msp_2 / total_tokens, total_l_flow / total_tokens


device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument('--train-mode', type=str, choices=['ae_only', 'flow_only', 'all'])
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--sigma', type=float, default=1.)
parser.add_argument('--sigma_decay_base', type=float, default=0.9)
parser.add_argument('--alpha-phase1', type=float, default=0.)
parser.add_argument('--alpha-phase2', type=float, default=0.)
parser.add_argument('--load-checkpoint', type=str, default=None)
args = parser.parse_args()

model_name = f'MolecularSimpleMSP_{args.task}_{args.train_mode}'
preds_path = os.path.join('preds', model_name)
save_path = os.path.join('saved', model_name)

if not os.path.exists(preds_path):
    os.makedirs(preds_path)
        
if not os.path.exists(save_path):
    os.makedirs(save_path)

    
# Get dataset
dataset, SRC, BOS_WORD, EOS_WORD, BLANK_WORD = get_dataset(args.task)
SOS_TOKEN = SRC.vocab.stoi[BOS_WORD]


# Define model
# encoder = EncoderRNN(input_size=len(SRC.vocab.itos), hidden_size=256, output_size=2048, n_layers=2)
# decoder = DecoderRNN(input_size=2048, hidden_size=256, output_size=len(SRC.vocab.itos), n_layers=2, sos_token=SOS_TOKEN)
# flow = FlowModel(2048, 4, 4, 256)
encoder = EncoderRNN(input_size=len(SRC.vocab.itos), hidden_size=256, output_size=100, n_layers=3)
decoder = DecoderRNN(input_size=100, hidden_size=256, output_size=len(SRC.vocab.itos), n_layers=3, sos_token=SOS_TOKEN)
vae = VAE(encoder, decoder).to(device)
# flow = FlowModel(100, 4, 4, 256)
flow = FlowModel(100, 6, 6, 256)
model = MSP(vae, flow, label_size=3).to(device)


if args.load_checkpoint is not None:
#     model.load_state_dict(torch.load(f'./saved/{args.load_checkpoint}/final_model.pt'), strict=False)
    model.vae.load_state_dict(torch.load(f'./saved/{args.load_checkpoint}/final_model.pt'), strict=False)

    
# Define optimizer
if args.train_mode == 'flow_only':
    model_params = model.flow.parameters()
else:
    model_params = model.parameters()
model_params = list(model_params)

optimizer = torch.optim.Adam(model_params, lr=args.lr)
criterion = nn.CrossEntropyLoss()


# Define data loaders
train_iter = data.BucketIterator(dataset=dataset, batch_size=256)
test_iter = data.BucketIterator(dataset=dataset, batch_size=32)


# Train model
for epoch in tqdm(range(1, args.epochs+1)): 
    # Training
    model.train()
    
    alpha = args.alpha

    if args.train_mode == "flow_only":
#         progress = (epoch - 50) / (args.epochs - 50)
        progress = epoch / args.epochs
        sigma = args.sigma * (args.sigma_decay_base ** progress)
    else:
        sigma = args.sigma
    
    loss, loss_rec, loss_vae, loss_msp_1, loss_msp_2, loss_flow = run_epoch(train_iter, model, optimizer, alpha, sigma)
    
    print(f"Epoch: {epoch}, loss = {loss}, loss rec = {loss_rec}, loss vae = {loss_vae}, loss msp 1 = {loss_msp_1}, loss msp 2 = {loss_msp_2}, loss flow = {loss_flow}")

    # Testing
    model.eval()
    
    # Test reconstruction
    batch = next(iter(test_iter))
    input = batch.smiles.to(device)
    _, _, _, _, decoded = model(input)
    
    input_smiles = return_input_smiles(batch.smiles.T, SRC)
    smiles = return_smiles(decoded, SRC)

    pd.DataFrame({
        'input_smiles': input_smiles,
        'reconstructed_smiles': smiles
    }).to_csv(f'{preds_path}/reconstruction_{epoch}.csv')

    # Test generation
#     z = torch.randn((16,2048)).to(device)
    z = torch.randn((16,100)).to(device)

    generated = model.vae.decoder.generate(z, n_steps=60, temperature=1.)
    smi_list = return_smiles(generated, SRC)
    
    pd.DataFrame({
        'generated_smiles': smi_list,
    }).to_csv(f'{preds_path}/generation_{epoch}.csv')
    
torch.save(model.state_dict(), f'{save_path}/final_model.pt')