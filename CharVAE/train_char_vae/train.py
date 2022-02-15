import os
import gin
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchtext import data

from src.vae import VAE, EncoderRNN, DecoderRNN
from src.loss import loss_vae
from src.data_utils import get_dataset
from src.utils import return_smiles, return_input_smiles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_epoch(data_iter, model, optimizer, criterion):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0

    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        optimizer.zero_grad()
        input = batch.smiles.to(device)
        
        m, l, z, decoded = model(input)
        
        _target = input.transpose(0, 1).flatten()
        _decoded = decoded.reshape(-1, decoded.size(2))

        ll_loss = criterion(_decoded, _target)
        kld_loss = loss_vae(m, l)
        loss = ll_loss + kld_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        total_tokens += batch.batch_size
            
    return total_loss / total_tokens


def test(vae, test_iter, batch_size, SRC, preds_path, epoch):
    vae.eval()

    # Test reconstruction
    batch = next(iter(test_iter))
    input = batch.smiles.to(device)
    _, _, _, decoded = vae(input)

    input_smiles = return_input_smiles(batch.smiles.T, SRC)
    smiles = return_smiles(decoded, SRC)

    pd.DataFrame({
        'input_smiles': input_smiles,
        'reconstructed_smiles': smiles
    }).to_csv(f'{preds_path}/reconstruction_{epoch}.csv')

    # Test generation
    z = torch.randn((batch_size, 128)).to(device)

    generated = vae.decoder.generate(z, n_steps=60, temperature=1.)
    smi_list = return_smiles(generated, SRC)

    pd.DataFrame({
        'generated_smiles': smi_list,
    }).to_csv(f'{preds_path}/generation_{epoch}.csv')


@gin.configurable
def train(encoder_fn, decoder_fn,
          num_epochs, lr, batch_size,
          model_name):
    preds_path = os.path.join('preds', model_name)
    save_path = os.path.join('saved', model_name)
    Path(preds_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    dataset, SRC, BOS_WORD, EOS_WORD, BLANK_WORD = get_dataset()
    SOS_TOKEN = SRC.vocab.stoi[BOS_WORD]

    encoder = encoder_fn(input_size=len(SRC.vocab.itos))
    decoder = decoder_fn(output_size=len(SRC.vocab.itos), sos_token=SOS_TOKEN)
    vae = VAE(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_iter = data.BucketIterator(dataset=dataset, batch_size=batch_size)
    test_iter = data.BucketIterator(dataset=dataset, batch_size=batch_size)

    for epoch in tqdm(range(num_epochs)):
        # Training
        vae.train()
        loss = run_epoch(train_iter, vae, optimizer, criterion)
        print(f"Epoch: {epoch}, loss = {loss}")

        # Testing
        test(vae, test_iter, batch_size, SRC, preds_path, epoch)

    torch.save(vae.state_dict(), f'{save_path}/final_model.pt')


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='configs/config.gin')

if __name__ == '__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.config_file)
    train()
