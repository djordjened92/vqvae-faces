import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from src.vqvae.model import VectorQuantizedVAE
from src.pixelCNN.model import PixelCNN, GatedPixelCNN, PixelSNAIL

class VQVAE_PixelCNN(nn.Module):
    def __init__(self, code_dim, code_size, latent_shape, pixelCNN_type, pixelCNN_n_layers=7):
        super().__init__()
        self.vqvae = VectorQuantizedVAE(code_dim, code_size).cuda()

        if pixelCNN_type in ['pixelCNN', 'gatedPixelCNN']:
            pixelCNN = pixelCNN_types[pixelCNN_type](
                input_shape=(input_h, input_w),
                size=code_size,
                n_layers=n_layers
            ).cuda()
        elif pixelCNN_type == 'pixelSNAIL':
            pixelCNN = PixelSNAIL(
                in_channels=1,
                out_channels=code_size,
                n_channels=code_size // 2,
                n_pixel_snail_blocks=n_layers,
                n_residual_blocks=2,
                attention_value_channels=code_size // 2,  # n_channels / 2
                attention_key_channels=16
            ).cuda()
        else:
            print(f'Incorrect pixelCNN model type')
            exit(1)
    
    def forward(self, x):
        # Encoder and Quantizier
        z = self.vqvae.encoder(x)
        e_q, e_st, e_ind = self.vqvae.codebook(z)

        # PixelCNN infernece
        e_ind = e_ind.unsqueeze(dim=1)
        e_ind_ar = self.pixelCNN(e_ind.detach())
        x_tilde = self.vqvae.decoder(e_st)
        commitment_loss = 0.15 * torch.mean((z - e_q) ** 2)
        return x_tilde, commitment_loss, e_ind, e_ind_ar
    
    def loss(self, x):
        x = 2 * x - 1
        x_tilde, commitment_loss, e_ind, e_ind_ar = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        ll = F.cross_entropy(e_ind_ar, e_ind.squeeze())
        loss = recon_loss + commitment_loss + 8*ll#(1e-2 if epoch > 30 else 0) * ll
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=commitment_loss, log_likelihood=ll)