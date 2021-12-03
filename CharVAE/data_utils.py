import pandas as pd
import torch
from torchtext import data, datasets
from SmilesPE.pretokenizer import atomwise_tokenizer


def get_dataset(task):
    if task == 'regression':
        path = './data/250k_rndm_zinc_drugs_clean_3_canonized.csv'
    elif task == 'regression_tpsa':
        path = './data/250k_tpsa.csv'
    elif task == 'regression_std':
        path = './data/250k_scaled.csv'
    elif task == 'classification':
        path = './data/250k_classification.csv'
    
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    SRC = data.Field(tokenize=atomwise_tokenizer, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    data_fields = [('smiles', SRC), 
                   ('logP', data.Field(is_target=True, use_vocab=False, sequential=False, preprocessing=float, dtype=torch.float)), 
                   ('qed', data.Field(is_target=True, use_vocab=False, sequential=False, preprocessing=float, dtype=torch.float)), 
                   ('SAS', data.Field(is_target=True, use_vocab=False, sequential=False, preprocessing=float, dtype=torch.float))]

    dataset = data.TabularDataset(path=path, format='csv', fields=data_fields, skip_header=True)

    SRC.build_vocab(dataset)
    return dataset, SRC, BOS_WORD, EOS_WORD, BLANK_WORD
