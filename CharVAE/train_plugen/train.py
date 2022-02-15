import os
import gin
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchtext import data

from src.vae import MSP, VAE, EncoderRNN, DecoderRNN
from src.flow import FlowModel
from src.loss import msp_vae_loss
from src.data_utils import get_dataset
from src.utils import return_smiles, return_input_smiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

def run_epoch(data_iter, model, optimizer, alpha, sigma, criterion, model_params, task):
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
        
        loss, l_rec, l_vae, l_msp_1, l_msp_2, l_flow = msp_vae_loss(criterion, input, decoded, mu, logvar, label, flow_z, task, alpha, sigma)

        # Flow only params
        (-1 * l_flow).backward()

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


def test(model, test_iter, batch_size, SRC, preds_path, epoch):
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
    z = torch.randn((batch_size, 100)).to(device)

    generated = model.vae.decoder.generate(z, n_steps=60, temperature=1.)
    smi_list = return_smiles(generated, SRC)

    pd.DataFrame({
        'generated_smiles': smi_list,
    }).to_csv(f'{preds_path}/generation_{epoch}.csv')


@gin.configurable
def train(encoder_fn, decoder_fn, flow_fn,
          task, num_features, load_checkpoint,
          num_epochs, lr, batch_size,
          alpha, sigma, sigma_decay_base,
          model_name):
    # Save paths
    preds_path = os.path.join('preds', model_name)
    save_path = os.path.join('saved', model_name)
    Path(preds_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Get dataset
    dataset, SRC, BOS_WORD, EOS_WORD, BLANK_WORD = get_dataset()
    SOS_TOKEN = SRC.vocab.stoi[BOS_WORD]

    # Define model
    encoder = encoder_fn(input_size=len(SRC.vocab.itos))
    decoder = decoder_fn(output_size=len(SRC.vocab.itos), sos_token=SOS_TOKEN)
    vae = VAE(encoder, decoder).to(device)
    flow = flow_fn()
    model = MSP(vae, flow, label_size=num_features).to(device)

    model.vae.load_state_dict(torch.load(load_checkpoint), strict=False)

    # Define optimizer
    model_params = model.flow.parameters()
    model_params = list(model_params)

    optimizer = torch.optim.Adam(model_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Define data loaders
    train_iter = data.BucketIterator(dataset=dataset, batch_size=batch_size)
    test_iter = data.BucketIterator(dataset=dataset, batch_size=batch_size)

    # Train model
    for epoch in tqdm(range(1, num_epochs+1)):
        # Training
        model.train()

        alpha = alpha

        progress = epoch / num_epochs
        sigma = sigma * (sigma_decay_base ** progress)

        loss, loss_rec, loss_vae, loss_msp_1, loss_msp_2, loss_flow = run_epoch(train_iter, model, optimizer, alpha, sigma, criterion, model_params, task)

        print(f"Epoch: {epoch}, loss = {loss}, loss rec = {loss_rec}, loss vae = {loss_vae}, loss msp 1 = {loss_msp_1}, loss msp 2 = {loss_msp_2}, loss flow = {loss_flow}")

        # Testing
        test(model, test_iter, batch_size, SRC, preds_path, epoch)

    torch.save(model.state_dict(), f'{save_path}/final_model.pt')


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='configs/config.gin')

if __name__ == '__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.config_file)
    train()
