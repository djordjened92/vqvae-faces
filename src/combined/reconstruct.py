import os
import json
import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from torchvision.utils import make_grid
from src.dataset import create_dataloader
from src.combined.combined_model import VQVAE_PixelCNN

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

def main(args):
    ckpt_path = args.model_path

    # Load configuration
    with open(os.path.join(os.path.dirname(ckpt_path), 'config.json'), 'r') as f:
        config = json.load(f)

    code_dim = config['CODE_DIM']
    code_size = config['CODE_SIZE']
    n_layers = config['NUM_OF_LAYERS']
    latent_shape = config['LATENT_SHAPE']
    test_path = config['TEST_PATH']

    # Load model
    vqvae_pixelCNN = VQVAE_PixelCNN(code_dim, code_size, (latent_shape[0], latent_shape[1]), config['GATED'], n_layers).cuda()
    checkpoint = torch.load(ckpt_path)
    vqvae_pixelCNN.load_state_dict(checkpoint['model_state_dict'])
    vqvae = vqvae_pixelCNN.vqvae

    # Load test data
    image_w = config['IMAGE_WIDTH']
    image_h = config['IMAGE_HEIGHT']
    test_dataloader = create_dataloader(test_path, image_w, image_h, batch_size=1, shuffle=True, workers=0)

    recs = reconstruction(vqvae, test_dataloader, 100)
    grid_img = make_grid(recs, nrow=20)
    img = grid_img.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize = (40, 40))
    img_name = os.path.basename(ckpt_path).split('.')[0]
    plt.imsave(f'recon_{img_name}.png', img)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, \
                        help='Path to vqvae model checkpoint')
    args = parser.parse_args()
    main(args)

'''
/home/djordje/Documents/Projects/vqvae-faces/checkpoints/combined/gated_model_12/gated_model_12_62.pt
'''