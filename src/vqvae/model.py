import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.RReLU(),
            nn.Conv2d(dim, dim, k, s, p),
            nn.BatchNorm2d(dim),
            nn.RReLU(),
            nn.Conv2d(dim, dim, 1)
        )
      
    def forward(self, x):
        return x + self.net(x)

class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(size, code_dim))
        self.register_buffer('N', torch.zeros(size))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        
        distances = ((flat_inputs.repeat(1, self.size).view(-1, self.size, self.code_dim) \
                     - self.embeddings) ** 2).sum(dim=-1)
        encoding_indices = torch.argmin(distances, dim=-1)
        encode_onehot = F.one_hot(encoding_indices, self.size).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = F.embedding(encoding_indices, self.embeddings).permute(0, 3, 1, 2).contiguous()

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.size * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

        return quantized, (quantized - z).detach() + z, encoding_indices

class VectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size
        self.code_dim = code_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.code_dim, 4, stride=2, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.Conv2d(self.code_dim, self.code_dim, 3, stride=2, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.Conv2d(self.code_dim, self.code_dim, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.Conv2d(self.code_dim, self.code_dim, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.Conv2d(self.code_dim, self.code_dim, 3, stride=2, padding=1),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim)
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, self.code_dim, 3, stride=2, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, self.code_dim, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, self.code_dim, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, self.code_dim, 3, stride=2, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        with torch.no_grad():
            x = 2 * x - 1
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = F.embedding(latents, self.codebook.embeddings).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents) * 0.5 + 0.5

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        commitment_loss = 0.15 * torch.mean((z - e) ** 2)
        return x_tilde, commitment_loss

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, diff = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        loss = recon_loss + diff
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)