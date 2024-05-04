import os
import torch
import numpy as np
from torch import nn
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles


def sigmoid_schedule(epoch, k=0.01, x0=29):
    """시그모이드 일정에 따라 가중치를 조절하는 함수."""
    return 1 / (1 + np.exp(-k * (epoch - x0)))

def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False

def sample_latent_space(vae_encoder, vae_decoder, sample_len, device):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension,
                                     device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)
        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms

def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld

def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality

def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding,
                         alphabet, sample_num, sample_len, device):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ''
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len, device):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(' ', '')

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)

def quality_in_valid_set(vae_encoder, vae_decoder, valid_loader, batch_size, device):
    num_batches_valid = len(valid_loader.dataset) // batch_size

    quality_list = []
    for batch in valid_loader:
        batch = batch[0]
        batch = batch.to(device)  # 입력 데이터를 GPU로 이동
        trg_len = batch.size(1)

        latent_points, mus, log_vars = vae_encoder(batch)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)

        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()

def quality_in_valid_set_QM9(vae_encoder, vae_decoder, data_valid, batch_size, device):
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        _, trg_len, _ = batch.size()

        latent_points, mus, log_vars = vae_encoder(batch)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()

def save_valid_samples_last_epoch(vae_encoder, vae_decoder, sample_num, sample_len, alphabet, file_name, device):
    valid_samples = []
    for _ in range(sample_num):
        sampled_atoms = sample_latent_space(vae_encoder, vae_decoder, sample_len)
        sampled_smiles = ''.join([alphabet[i] for i in sampled_atoms]).replace(' ', '')
        if is_correct_smiles(sampled_smiles):
            valid_samples.append(sampled_smiles)

    with open(file_name, 'w') as file:
        for smiles in valid_samples:
            file.write(smiles + '\n')