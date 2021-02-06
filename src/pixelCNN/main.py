import os
import json
import shutil
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset import create_dataloader
from src.training import train_epochs
from model import PixelCNN
from src.vqvae.model import VectorQuantizedVAE

def create_prior_dataset(data_loader, model):
    prior_data = []
    model.eval()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            z = model.cuda().encode_code(x)
            prior_data.append(z.long())
    return torch.cat(prior_data, dim=0)

#########################################################
# Load VQVAE model
# Get checkpoint of trained vqvae model
ckpt_step = 320
vqvae_model_name = 'model_3'
vqvae_model_path = f'../../checkpoints/vqvae/{vqvae_model_name}'

# Load configuration
with open(f'{vqvae_model_path}/config.json', 'r') as f:
    vqvae_config = json.load(f)

vqvae_ckpt_path = os.path.join(vqvae_model_path, f'{vqvae_model_name}_{ckpt_step}.pt')
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
with open(f'config.json', 'r') as f:
    config = json.load(f)

# Create checkpoint directory
ckpt_path = os.path.join(config["CKPT_PATH"], config["MODEL_NAME"])
Path(ckpt_path).mkdir(parents=True, exist_ok=True)
# Copy current configuration to model's directory
shutil.copy('config.json', ckpt_path)

# Create datasets
train_path = vqvae_config['TRAIN_PATH']
valid_path = vqvae_config['VALIDATION_PATH']
input_w = config['INPUT_WIDTH']
input_h = config['INPUT_HEIGHT']
batch_size = config['BATCH_SIZE']
shuffle = config['SHUFFLE']
workers = config['NUM_WORKERS']
code_dim = config['CODE_DIM']
code_size = config['CODE_SIZE']
n_layers = config['NUM_OF_LAYERS']

train_dataloader = create_dataloader(train_path, image_w, image_h, batch_size, shuffle, workers)
val_dataloader = create_dataloader(valid_path, image_w, image_h, batch_size, shuffle, workers)
prior_train_data, prior_val_data = create_prior_dataset(train_dataloader, vqvae), create_prior_dataset(val_dataloader, vqvae)
prior_train_loader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=True)
prior_val_loader = DataLoader(prior_val_data, batch_size=batch_size)

# Instantiate model
pixelCNN = PixelCNN(code_size=code_size, input_shape=(input_h, input_w), dim=code_dim, n_layers=n_layers).cuda()

# Define TB writer
writer = SummaryWriter(f'{config["LOGS_PATH"]}/{config["MODEL_NAME"]}')

# Run training
train_args = {
    'epochs': config['EPOCHS'],
    'lr': config['LR'],
    'grad_clip': config['GRAD_CLIP'],
    'ckpt_period': config['CKPT_PERIOD'],
    'ckpt_path': ckpt_path,
    'model_name': config['MODEL_NAME']
}
train_epochs(pixelCNN, prior_train_loader, prior_val_loader, train_args, writer, quiet=False)
writer.close()