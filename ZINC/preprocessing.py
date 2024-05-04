import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_smiles_encodings_for_dataset(file_path):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES, 
    given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))


    return smiles_list, smiles_alphabet, largest_smiles_len


def one_hot_to_smiles(one_hot_encoded, alphabet):
    """
    원-핫 인코딩된 분자를 SMILES 문자열로 변환합니다.

    Args:
    - one_hot_encoded (torch.Tensor): 원-핫 인코딩된 분자 데이터
    - alphabet (list): SMILES 알파벳 리스트

    Returns:
    - str: SMILES 문자열
    """
    # 원-핫 인코딩에서 가장 높은 값의 인덱스를 찾아 해당 알파벳 문자로 변환
    smiles_idx = torch.argmax(one_hot_encoded, dim=-1)
    smiles_str = ''.join([alphabet[i] for i in smiles_idx])

    # 불필요한 공백 문자 제거
    smiles_str = smiles_str.replace(' ', '')
    return smiles_str


def smile_to_hot(smile, largest_smile_len, alphabet):
    """Go from a single smile string to a one-hot encoding."""
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    smile += ' ' * (largest_smile_len - len(smile))
    integer_encoded = [char_to_int[char] for char in smile]
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return np.array(onehot_encoded)


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """Convert a list of smile strings to a one-hot encoding."""
    hot_list = []
    for s in tqdm(smiles_list, desc="Encoding SMILES"):
        onehot_encoded = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)
