# 💊 Sequence-based Molecular Generation using VAE

## Overview
This project focuses on creating new molecules using sequence-based molecular datasets. The primary goal is to improve the performance of molecular generation by modifying the model's architecture and learning methodologies.

### Datasets (Let's create a new molecule using a sequence-based molecule dataset!)
- **ZINC 250k dataset**
  - A curated collection of over 250,000 commercially available chemical compounds, commonly used in drug discovery projects.
  - Compounds are represented in SMILES format, which encodes the structure of molecules for various computational chemistry tasks including virtual screening and molecule generation.
  - This dataset is notable for its diversity in chemical space, which is crucial for training models to predict molecular properties or to generate new molecules.

- **QM9 dataset**
  - Consists of approximately 134,000 molecules with up to 9 heavy atoms (C, O, N, F), excluding hydrogen.
  - Each molecule is represented in SMILES format.
  - Includes extensive quantum mechanical properties calculated at a high level of theory (DFT/B3LYP), making it useful for training and benchmarking models that predict quantum mechanical properties of small organic molecules.

## 🔥 Project Goals (No biology knowledge required!)
- **Improve the validity and diversity of generated molecules from latent space.**
  - **Validity:** Ensuring that the molecules generated are chemically valid.
  - **Diversity:** Generating a wide range of different molecules.
- **Data Monitoring:**
  - `smiles_pairs_per_epoch.dat`: Extract and compare three random molecules each epoch for Input and reconstruction analysis.
  - `smiles_sampling_per_epoch.dat`: Extract 10 valid molecules from the latent space each epoch.

## 🌍 Installation

Create a virtual environment and install the required packages (Python version: 3.7):

```bash
pip install rdkit-pypi
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyyaml
pip install scikit-learn
pip install pandas
pip install tqdm
```

## 📞 Contact

For further information or queries, please contact:
- Email: h_j_song@korea.ac.kr
