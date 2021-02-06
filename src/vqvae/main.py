import os
import json
import shutil
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from src.dataset import create_dataloader
from src.training import train_epochs
from model import VectorQuantizedVAE

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Create checkpoint directory
ckpt_path = os.path.join(config["CKPT_PATH"], config["MODEL_NAME"])
Path(ckpt_path).mkdir(parents=True, exist_ok=True)
# Copy current configuration to model's directory
shutil.copy('config.json', ckpt_path)

# Create datasets
train_path = config['TRAIN_PATH']
valid_path = config['VALIDATION_PATH']
test_path = config['TEST_PATH']
image_w = config['IMAGE_WIDTH']
image_h = config['IMAGE_HEIGHT']
batch_size = config['BATCH_SIZE']
shuffle = config['SHUFFLE']
workers = config['NUM_WORKERS']

train_dataloader = create_dataloader(train_path, image_w, image_h, batch_size, shuffle, workers)
val_dataloader = create_dataloader(valid_path, image_w, image_h, batch_size, workers=workers)

# Instantiate model
code_dim = config['CODE_DIM']
code_size = config['CODE_SIZE']
vqvae = VectorQuantizedVAE(code_dim, code_size).cuda()

# Define TB writer
writer = SummaryWriter(f'{config["LOGS_PATH"]}/{config["MODEL_NAME"]}')
writer.add_graph(vqvae, iter(train_dataloader).next().cuda())

# Run training
train_args = {
    'epochs': config['EPOCHS'],
    'lr': config['LR'],
    'grad_clip': config['GRAD_CLIP'],
    'ckpt_period': config['CKPT_PERIOD'],
    'ckpt_path': ckpt_path,
    'model_name': config['MODEL_NAME']
}
train_epochs(vqvae, train_dataloader, val_dataloader, train_args, writer, quiet=False)
writer.close()