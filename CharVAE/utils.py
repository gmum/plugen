import torch
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem


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
        smi = smi.rstrip('</s')
        smi = smi.rstrip('</')
        smi = smi.rstrip('<')
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


def fingerprint_similarity(fp1, fp2):
    """Calculate fingerprint Tanimoto similarity using CUDA
    :param fp1: list of lists or numpy.array; Single fingerprint or a list of fingerprints to compare
    :param fp2: list of lists or numpy.array; Single fingerprint or a list of fingerprints to compare to
    :return: numpy.array; Array of Tanimoto distances of fingerprints
    """
    fp1 = torch.tensor(fp1, dtype=torch.float32).cuda()
    fp2 = torch.tensor(fp2, dtype=torch.float32).cuda()
    mul = torch.matmul(fp1, fp2.t()).float()
    sum1 = torch.sum(fp1, dim=1)
    sum2 = torch.sum(fp2, dim=1)
    del fp1, fp2
    r_sum1 = sum1.repeat(sum2.shape[0], 1).float()
    r_sum2 = sum2.repeat(sum1.shape[0], 1).float()
    del sum1, sum2
    output = mul/((r_sum2+r_sum1.t())-mul)
    final_output = output.cpu().numpy()
    del output, mul, r_sum1, r_sum2
    torch.cuda.empty_cache()
    return final_output


def ecfp_from_mol(mol, radius=3, nBits=1024):
    """
Computes ECFP fingerprint from a SMILES
    :param smi: str; SMILES of a molecule
    :param radius: int; Radius of ECFP fingerprint
    :param nBits: int; Number of bits for the final FP
    :return: list of ints; Final fingerprint as a list of ints
    """
    return list(AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nBits,
        useChirality=False,
        useBondTypes=True,
        useFeatures=False)
    )