import os
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import create_dataloader
from src.training import train_epochs
from src.pixelCNN.model import PixelCNN, GatedPixelCNN, PixelSNAIL
from src.vqvae.model import VectorQuantizedVAE

# Try to catch cause of anomalies like nan loss
torch.autograd.set_detect_anomaly(True)

def main(args):
    # Load pixelcnn configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    def create_prior_dataset(data_loader, model):
        prior_data = []
        model.eval()
        with torch.no_grad():
            for x in data_loader:
                x = x.cuda()
                z = model.cuda().encode_code(x)
                z = z.unsqueeze(dim=0)
                prior_data.append(z.long())
        return torch.cat(prior_data, dim=0)

    #########################################################
    # Load VQVAE model

    vqvae_ckpt_path = config['VQVAE_MODEL']

    # Load configuration
    with open(os.path.join(os.path.dirname(vqvae_ckpt_path), 'config.json'), 'r') as f:
        vqvae_config = json.load(f)

    vqvae_code_dim = vqvae_config['CODE_DIM']
    vqvae_code_size = vqvae_config['CODE_SIZE']
    image_h = vqvae_config['IMAGE_HEIGHT']
    image_w = vqvae_config['IMAGE_WIDTH']

    # Load model
    vqvae = VectorQuantizedVAE(vqvae_code_dim, vqvae_code_size).cuda()
    checkpoint = torch.load(vqvae_ckpt_path)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    #########################################################
    # Train PixelCNN model
    # Create checkpoint directory
    config["MODEL_NAME"] = config['TYPE'] + '_' + config["MODEL_NAME"]

    ckpt_path = os.path.join(config["CKPT_PATH"], config["MODEL_NAME"])
    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    # Copy current configuration to model's directory
    shutil.copy(args.config_path, ckpt_path)

    # Create datasets
    train_path = vqvae_config['TRAIN_PATH']
    valid_path = vqvae_config['VALIDATION_PATH']
    input_w = config['INPUT_WIDTH']
    input_h = config['INPUT_HEIGHT']
    batch_size = config['BATCH_SIZE']
    shuffle = config['SHUFFLE']
    workers = config['NUM_WORKERS']
    code_size = config['CODE_SIZE']
    n_layers = config['NUM_OF_LAYERS']
    pretrained_path = config['PRETRAINED']

    train_dataloader = create_dataloader(train_path, image_w, image_h, 1, False, workers)
    val_dataloader = create_dataloader(valid_path, image_w, image_h, 1, False, workers)
    
    # Load prior dataset either from numpy files, or generating it from images
    if args.save_prior_ds == 'True':
        prior_train_data = create_prior_dataset(train_dataloader, vqvae)
        prior_val_data = create_prior_dataset(val_dataloader, vqvae)
        np.save(args.train_set_path, prior_train_data.cpu())
        np.save(args.valid_set_path, prior_val_data.cpu())
    else:
        prior_train_data = np.load(args.train_set_path) if args.train_set_path else create_prior_dataset(train_dataloader, vqvae)
        prior_val_data = np.load(args.valid_set_path) if args.valid_set_path else create_prior_dataset(val_dataloader, vqvae)

    prior_train_loader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=shuffle)
    prior_val_loader = DataLoader(prior_val_data, batch_size=batch_size)

    # Instantiate model
    pixelCNN_types = {
        'pixelCNN': PixelCNN,
        'gatedPixelCNN': GatedPixelCNN
    }

    pixelCNN_type = config['TYPE']
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
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path)
        pixelCNN.load_state_dict(checkpoint['model_state_dict'])
    
    # Define TB writer
    writer = SummaryWriter(f'{config["LOGS_PATH"]}/{config["MODEL_NAME"]}')

    # Run training
    train_args = {
        'epochs': config['EPOCHS'],
        'lr': config['LR'],
        'grad_clip': config['GRAD_CLIP'],
        'ckpt_period': config['CKPT_PERIOD'],
        'weight_decay': config['WEIGHT_DECAY'],
        'ckpt_path': ckpt_path,
        'model_name': config['MODEL_NAME'],
        'lr_scheduler': config['LR_SCHEDULER'],
        'optimizer': config['OPTIMIZER']
    }
    train_epochs(pixelCNN, prior_train_loader, prior_val_loader, train_args, writer, quiet=False)
    writer.close()
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, \
                        help='Path to configuration file for current PixelCNN training')
    parser.add_argument('--train_set_path', type=str, \
                        help='Path to training numpy structure')
    parser.add_argument('--valid_set_path', type=str, \
                        help='Path to validation numpy structure')
    parser.add_argument('--save_prior_ds', type=str, \
                        help='Flag for saving generated prior dataset')
    args = parser.parse_args()
    main(args)