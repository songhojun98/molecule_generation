import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils import *
from preprocessing import one_hot_to_smiles

def train_model(vae_encoder, vae_decoder,
                train_loader, valid_loader, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding,
                step_size, gamma, results_dir, device):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    # initialize a learning rate scheduler
    scheduler_encoder = StepLR(optimizer_encoder, step_size=step_size, gamma=gamma)
    scheduler_decoder = StepLR(optimizer_decoder, step_size=step_size, gamma=gamma)

    best_corr = -1.0  # 초기 최고 corr 값 설정
    input_output_pairs = []

    quality_valid_list = [0, 0, 0, 0]
    
    total_batches = len(train_loader.dataset) // batch_size
    if len(train_loader.dataset) % batch_size != 0:
        total_batches += 1

    for epoch in range(num_epochs):

        start = time.time()
        
        # DataLoader를 통해 학습 데이터의 배치 처리
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)  # 배치 데이터를 모델이 있는 장치로 이동
            

            latent_points, mus, log_vars = vae_encoder(batch)
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()


            if batch_idx % 30 == 0:
                end = time.time()  # 현재 시간

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(vae_encoder, vae_decoder, valid_loader, batch_size, device)
                report = 'Epoch: %d,  Batch: %d / %d, \t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_idx, total_batches, loss.item(), quality_train, quality_valid, end - start)
                print(report)
                start = time.time()  # 시간 재설정

        # 유효한 SMILES 샘플 저장을 위한 코드 수정
        valid_smiles_samples = []
        attempts = 0
        while len(valid_smiles_samples) < 10 and attempts < 100:
            sampled_atoms = sample_latent_space(vae_encoder, vae_decoder, sample_len, device)
            sampled_smiles = ''.join([alphabet[i] for i in sampled_atoms]).replace(' ', '')
            if is_correct_smiles(sampled_smiles):
                valid_smiles_samples.append(sampled_smiles)
            attempts += 1

        # 유효한 SMILES 샘플을 .dat 파일에 저장 (기존 .txt 파일 대신)
        with open(results_dir + '/smiles_sampling_per_epoch.dat', 'a') as file:
            file.write(f"Epoch {epoch}:\n")
            for smiles in valid_smiles_samples:
                file.write(smiles + '\n')
            file.write("\n")  # 에폭 간 구분을 위한 줄바꿈 추가

        # 에포크 종료 후 유효성 검사
        quality_valid = quality_in_valid_set(vae_encoder, vae_decoder, valid_loader, batch_size, device)
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder, type_of_encoding, alphabet, sample_num, sample_len, device)
            report = f'Epoch: {epoch} | Validity: {corr * 100. / sample_num:.5f} % | Diversity: {unique * 100. / sample_num:.5f} % | Reconstruction: {quality_valid:.5f} %'
            print(report)
            with open(results_dir + '/results.dat', 'a') as content:
                content.write(report + '\n')

        # 초기 종료 조건 확인
        if quality_valid_list[-1] < 70. and epoch > 200:
            print('Early stopping')
            break

        
        scheduler_encoder.step(quality_valid)
        scheduler_decoder.step(quality_valid)

        # 임의의 분자 10개 선택 및 처리
        random_batch_idx = torch.randint(0, len(train_loader.dataset), (1,)).item()
        random_batch = next(iter(DataLoader(dataset=train_loader.dataset, batch_size=3, sampler=torch.utils.data.SubsetRandomSampler([random_batch_idx]))))
        random_samples = random_batch[0].to(device)  # 배치 데이터를 모델이 있는 장치로 이동

        input_smiles_list = []
        output_smiles_list = []

        for sample in random_samples:
            latent_points, mus, log_vars = vae_encoder(sample.unsqueeze(0))
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=1)
            out_one_hot = torch.zeros_like(sample.unsqueeze(0), device=device)

            for seq_index in range(sample.shape[0]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            input_smiles = one_hot_to_smiles(sample, alphabet)
            output_smiles = one_hot_to_smiles(out_one_hot[0], alphabet)

            input_smiles_list.append(input_smiles)
            output_smiles_list.append(output_smiles)

        # 파일에 SMILES 쌍 저장
        with open(results_dir + "/smiles_pairs_per_epoch.dat", 'a') as file:
            file.write(f"Epoch {epoch}\nInput:\n")
            file.writelines([f"{smiles}\n" for smiles in input_smiles_list])
            file.write("Output:\n")
            file.writelines([f"{smiles}\n" for smiles in output_smiles_list])
            file.write("\n")

        # 리스트 초기화
        input_output_pairs = []

        # [에포크 종료 후 평가 및 모델 저장 처리 코드]

        scheduler_encoder.step()
        scheduler_decoder.step()