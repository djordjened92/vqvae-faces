import os
import json
from torch import nn
from src.vqvae.model import VectorQuantizedVAE
from src.pixelCNN.model import PixelCNN, GatedPixelCNN

class VQVAE_PixelCNN(nn.Module):
    def __init__(self, code_dim, code_size, latent_shape, pixelCNN_gated=False, pixelCNN_n_layers=7):
        super().__init__()
        self.vqvae = VectorQuantizedVAE(code_dim, code_size).cuda()

        if pixelCNN_gated:
            self.pixelCNN = GatedPixelCNN(latent_shape, size=code_size, n_layers=pixelCNN_n_layers).cuda()
        else:
            self.pixelCNN = PixelCNN(latent_shape, size=code_size, n_layers=pixelCNN_n_layers).cuda()
    
    def forward(self, x):
        # Encoder and Quantizier
        z = self.vqvae.encoder(x)
        _, _, e_ind = self.vqvae.codebook(z)

        # PixelCNN infernece
        e_ind_ar = self.pixelCNN(e_ind)
        e = self.embedding(e_ind_ar).permute(0, 3, 1, 2).contiguous()
        e_st = (e - z).detach + z
        x_tilde = self.vqvae.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        return x_tilde, diff1 + diff2, e_ind, e_ind_ar
    
    def loss(self, x):
        x = 2 * x - 1
        x_tilde, diff, e_ind, e_ind_ar = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        ll = F.cross_entropy(e_ind, e_ind_ar)
        loss = recon_loss + diff + ll
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff, log_likelihood=ll)
