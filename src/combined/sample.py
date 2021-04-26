import os
import json
import shutil
import torch
import numpy as np
from skimage import io
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from combined_model import VQVAE_PixelCNN

def main(args):
    combined_ckpt_path = args.model_ckpt_path
    num_samples = args.num_samples
    samples_dir = args.samples_dir

    with open(os.path.join(os.path.dirname(combined_ckpt_path), 'config.json'), 'r') as f:
        config = json.load(f)

    # Instantiate model
    model_name = config['MODEL_NAME']
    code_dim = config['CODE_DIM']
    code_size = config['CODE_SIZE']
    n_layers = config['NUM_OF_LAYERS']
    latent_shape = config['LATENT_SHAPE']

    vqvae_pixelCNN = VQVAE_PixelCNN(code_dim, code_size, (latent_shape[0], latent_shape[1]), config['GATED'], n_layers).cuda()
    checkpoint = torch.load(combined_ckpt_path)
    vqvae_pixelCNN.load_state_dict(checkpoint['model_state_dict'])

    # Sample images
    vqvae_pixelCNN.eval()
    vqvae_pixelCNN.vqvae.eval()
    vqvae_pixelCNN.pixelCNN.eval()

    samples = vqvae_pixelCNN.pixelCNN.sample(num_samples).long()
    samples = (vqvae_pixelCNN.vqvae.decode_code(samples).permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)

    # Save sampled images
    samples_dir = os.path.join(samples_dir, model_name)
    Path(samples_dir).mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        path = os.path.join(samples_dir, f'{i}.jpg')
        io.imsave(path, sample)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_ckpt_path', type=str, required=True, \
                        help='Path to VQVAE_PixelCNN model checkpoint')
    parser.add_argument('--num_samples', type=int, required=True, \
                        help='Number of images to be sampled')
    parser.add_argument('--samples_dir', type=str, required=True, \
                        help='Path to directory where samples should be stored')
    args = parser.parse_args()
    main(args)

'''
python combined/sample.py --model_ckpt_path='/home/djordje/Documents/Projects/vqvae-faces/checkpoints/combined/model_2/model_2_6.pt' \
    --num_samples=100 --samples_dir='/home/djordje/Documents/Projects/vqvae-faces/samples'
'''