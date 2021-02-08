import os
import torch
import json
import numpy as np
from skimage import io
from argparse import ArgumentParser
from src.vqvae.model import VectorQuantizedVAE
from src.pixelCNN.model import PixelCNN

torch.manual_seed(123)

def main(args):
    vqvae_ckpt_path = args.vqvae_path
    pixelcnn_model_path = args.pixelcnn_path
    num_samples = args.num_samples
    samples_dir = args.samples_dir

    # Load both models' configs
    with open(os.path.join(os.path.dirname(vqvae_ckpt_path), 'config.json'), 'r') as f:
        vqvae_config = json.load(f)
    
    with open(os.path.join(os.path.dirname(pixelcnn_model_path), 'config.json'), 'r') as f:
        pixelcnn_config = json.load(f)

    # Instatiate models
    code_dim = vqvae_config['CODE_DIM']
    code_size = vqvae_config['CODE_SIZE']
    
    vqvae = VectorQuantizedVAE(code_dim, code_size).cuda()
    checkpoint = torch.load(vqvae_ckpt_path)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    input_w = pixelcnn_config['INPUT_WIDTH']
    input_h = pixelcnn_config['INPUT_HEIGHT']
    code_dim = pixelcnn_config['CODE_DIM']
    code_size = pixelcnn_config['CODE_SIZE']
    n_layers = pixelcnn_config['NUM_OF_LAYERS']

    pixelCNN = PixelCNN(code_size=code_size, input_shape=(input_h, input_w), dim=code_dim, n_layers=n_layers).cuda()
    checkpoint = torch.load(pixelcnn_model_path)
    pixelCNN.load_state_dict(checkpoint['model_state_dict'])

    # Sample images
    vqvae.eval()
    pixelCNN.eval()
    samples = pixelCNN.sample(num_samples).long()
    samples = (vqvae.decode_code(samples).permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)

    for i, sample in enumerate(samples):
        path = os.path.join(samples_dir, f'{i}.jpg')
        io.imsave(path, sample)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vqvae_path', type=str, required=True, \
                        help='Path to vqvae model checkpoint')
    parser.add_argument('--pixelcnn_path', type=str, required=True, \
                        help='Path to pixelCNN model checkpoint')
    parser.add_argument('--num_samples', type=int, required=True, \
                        help='Number of images to be sampled')
    parser.add_argument('--samples_dir', type=str, required=True, \
                        help='Path to directory where samples should be stored')
    args = parser.parse_args()
    main(args)