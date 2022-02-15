def print_smiles(tensor, SRC, count=4):
    for ex in tensor.argmax(-1).data.tolist()[:count]:
        smi = ''.join(list(map(lambda x: SRC.vocab.itos[x], ex)))
        smi = smi.lstrip('<s>')
        smi = smi.rstrip('<blank>')
        smi = smi.rstrip('</s>')
        print(smi)
    print('=====')
    
    
def return_smiles(tensor, SRC):
    smi_list = []
    for ex in tensor.argmax(-1).data.tolist():
        smi = ''.join(list(map(lambda x: SRC.vocab.itos[x], ex)))
        smi = smi.lstrip('<s>')
        smi = smi.rstrip('<blank>')
#         smi = smi.rstrip('</s>')
        where_end = smi.find('</s>')
        if where_end != -1:
            smi = smi[:where_end]
        smi_list.append(smi)
    return smi_list


def return_input_smiles(tensor, SRC):
    smi_list = []
    for ex in tensor.data.tolist():
        smi = ''.join(list(map(lambda x: SRC.vocab.itos[x], ex)))
        smi = smi.lstrip('<s>')
        smi = smi.rstrip('<blank>')
        smi = smi.rstrip('</s>')
        smi_list.append(smi)
    return smi_list