import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from torchvision.utils import make_grid
from src.dataset import create_dataloader
from src.training import train_epochs
from src.vqvae.model import VectorQuantizedVAE

torch.manual_seed(123)

def reconstruction(vqvae, data_loader, images_cnt = 2):
    test_iterator = iter(data_loader)
    out = []
    for i in range(images_cnt):
        x = next(test_iterator).cuda()
        vqvae.eval()
        with torch.no_grad():
            z = vqvae.encode_code(x)
            x_recon = vqvae.decode_code(z)
        out.append(torch.cat((x, x_recon), axis=0))

    out = torch.cat(out, axis=0)
    return out

# Get checkpoint of trained model
ckpt_step = 320
model_name = 'model_3'
model_path = f'../../checkpoints/vqvae/{model_name}'

# Load configuration
with open(f'{model_path}/config.json', 'r') as f:
    config = json.load(f)

ckpt_path = os.path.join(model_path, f'{model_name}_{ckpt_step}.pt')
code_dim = config['CODE_DIM']
code_size = config['CODE_SIZE']
test_path = config['TEST_PATH']

# Load model
vqvae = VectorQuantizedVAE(code_dim, code_size).cuda()
checkpoint = torch.load(ckpt_path)
vqvae.load_state_dict(checkpoint['model_state_dict'])

# Load test data
image_w = config['IMAGE_WIDTH']
image_h = config['IMAGE_HEIGHT']
test_dataloader = create_dataloader(test_path, image_w, image_h, batch_size=1, shuffle=True, workers=0)

recs = reconstruction(vqvae, test_dataloader, 100)
grid_img = make_grid(recs, nrow=20)
img = grid_img.permute(1, 2, 0).cpu().numpy()
plt.figure(figsize = (40, 40))
plt.imsave('reconstructions.png', img)