import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
      
    def forward(self, x):
        return x + self.net(x)

class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        
        distances = ((flat_inputs.repeat(1, self.size).view(-1, self.size, self.code_dim) \
                     - weight) ** 2).sum(dim=-1)
        encoding_indices = torch.min(distances, dim=-1)[1]
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices

class VectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size
        self.code_dim = code_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, max(self.code_dim // 4, 1), 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(max(self.code_dim // 4, 1)),
            nn.Conv2d(max(self.code_dim // 4, 1), max(self.code_dim // 2, 1), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(max(self.code_dim // 2, 1)),
            nn.Conv2d(max(self.code_dim // 2, 1), self.code_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.code_dim),
            nn.Conv2d(self.code_dim, self.code_dim, 4, stride=2, padding=1),
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(self.code_dim),
            ResidualBlock(self.code_dim),
            nn.ReLU(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, self.code_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.code_dim),
            nn.ConvTranspose2d(self.code_dim, max(self.code_dim // 2, 1), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(max(self.code_dim // 2, 1)),
            nn.ConvTranspose2d(max(self.code_dim // 2, 1), max(self.code_dim // 4, 1), 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(max(self.code_dim // 4, 1)),
            nn.ConvTranspose2d(max(self.code_dim // 4, 1), 3, 4, stride=2, padding=1),
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
            latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents) * 0.5 + 0.5

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        return x_tilde, diff1 + diff2

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, diff = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        loss = recon_loss + diff
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)