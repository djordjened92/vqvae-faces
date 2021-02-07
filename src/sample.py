import os
import torch
import json
from argparse import ArgumentParser
from src.vqvae.model import VectorQuantizedVAE
from src.pixelCNN.model import PixelCNN

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

    # Sample images


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arg1', type=int)
    args = parser.parse_args()
    main(args)