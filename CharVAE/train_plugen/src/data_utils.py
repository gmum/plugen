import gin
import torch
from torchtext import data, datasets
from SmilesPE.pretokenizer import atomwise_tokenizer


@gin.configurable
def get_dataset(data_path, features):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    SRC = data.Field(tokenize=atomwise_tokenizer, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    data_fields = [('smiles', SRC)] + \
                  [(feature, data.Field(is_target=True, use_vocab=False, sequential=False, preprocessing=float, dtype=torch.float)) for feature in features]

    dataset = data.TabularDataset(path=data_path, format='csv', fields=data_fields, skip_header=True)

    SRC.build_vocab(dataset)
    return dataset, SRC, BOS_WORD, EOS_WORD, BLANK_WORD
