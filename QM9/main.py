import os
import yaml
import argparse
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from train import train_model
from models import VAEEncoder, VAEDecoder
from preprocessing import get_smiles_encodings_for_dataset, multiple_smile_to_hot

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def main():
    if os.path.exists("settings.yml"):
        settings = yaml.safe_load(open("settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']
    lr_enc = training_parameters['lr_enc']
    KLD_alpha = training_parameters['KLD_alpha']
    latent_dimension = settings['encoder']['latent_dimension']

    # 결과 파일을 저장할 디렉토리 경로 설정
    results_dir = f'results/results_{batch_size}_{KLD_alpha}_{latent_dimension}_{lr_enc}'
    os.makedirs(results_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # 파일 경로 수정
    log_file_path = os.path.join(results_dir, 'logfile.dat')
    results_file_path = os.path.join(results_dir, 'results.dat')
    recon_file_path = os.path.join(results_dir, 'smiles_pairs_per_epoch.dat')
    sampling_file_path = os.path.join(results_dir, 'smiles_sampling_per_epoch.dat')

    # 파일 초기화
    content = open(log_file_path, 'w')
    content.close()
    content = open(results_file_path, 'w')
    content.close()
    content = open(recon_file_path, 'w')
    content.close()
    content = open(sampling_file_path, 'w')
    content.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')


    print('Representation: SMILES')
    encoding_list, encoding_alphabet, largest_molecule_len = get_smiles_encodings_for_dataset(file_name_smiles)

    print('--> Creating one-hot encoding...')
    data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                    encoding_alphabet)
    print('Finished creating one-hot encoding.')


    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")

    vae_encoder = VAEEncoder(in_dimension=len_max_molec, latent_dimension=latent_dimension).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)

    print('*' * 15, ': -->', device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.9, 0.1, 0.0]
    data = data[torch.randperm(data.size()[0])]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                alphabet=encoding_alphabet,
                type_of_encoding=type_of_encoding,
                sample_len=len_max_molec,
                device=device,
                results_dir=results_dir
                )

    with open('COMPLETED', 'w') as content:
        content.write('exit code: 0')



if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)