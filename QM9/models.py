import torch
import torch.nn.functional as F
from torch import nn

class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, latent_dimension):
        self.latent_dimension = latent_dimension
        super(VAEEncoder, self).__init__()

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Conv1d(in_dimension, 5, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(5, 4, kernel_size=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(48, 240)

        # Latent space mean
        self.encode_mu = nn.Linear(240, latent_dimension)
        # Latent space variance
        self.encode_log_var = nn.Linear(240, latent_dimension)

        self.relu = nn.ReLU()
        
    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        x = self.encode_nn(x) # Convolutional layers
        x = self.flatten(x) # Flatten layer
        x = F.selu(self.linear_1(x))  # Dense layer

        # latent space
        mu = self.encode_mu(x)
        log_var = self.encode_log_var(x)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        """
        Through Decoder
        """
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension)
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden