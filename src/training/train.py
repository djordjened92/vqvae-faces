import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from collections import OrderedDict
from tqdm import tqdm

optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'sparse_adam': optim.SparseAdam,
    'adamw': optim.AdamW,
    'adagrad': optim.Adagrad,
    'rmsprop': optim.RMSprop,
    'asgd': optim.ASGD
}

def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        x = x.cuda()
        out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda()
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Validation '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses


def train_epochs(model, train_loader, val_loader, train_args, tb_writer, quiet=False):
    epochs, lr = train_args['epochs'], train_args['lr']
    weight_decay = train_args.get('weight_decay', 0)
    grad_clip = train_args.get('grad_clip', None)
    optim_param = train_args.get('optimizer', 'adam')
    optimizer = optimizers[optim_param](model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = train_args.get('lr_scheduler', None)
    def lambda_lr(epoch):
        if epoch <= 30:
            return 1.
        elif 30 < epoch <= 60:
            return 0.5
        elif 60 < epoch <= 120:
            return 0.25
        elif 120 < epoch <= 200:
            return 0.125
        else:
            return 0.0125

    if lr_scheduler == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
    elif lr_scheduler == 'lambda_lr':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif lr_scheduler == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        val_loss = eval_loss(model, val_loader, quiet)
        
        # Save checkpoints
        if epoch % train_args['ckpt_period'] == 0:
            ckpt_path = f'{train_args["ckpt_path"]}/{train_args["model_name"]}_{epoch}.pt'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, ckpt_path)
        # Write losses to tensorboard
        for k in train_loss.keys():
            tb_writer.add_scalar(f'training/{k}', np.mean(train_loss[k]), epoch)
            tb_writer.add_scalar(f'validation/{k}', val_loss[k], epoch)
        
        # Apply lr_scheduler
        if lr_scheduler:
            scheduler.step()