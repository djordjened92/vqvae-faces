

# Instantiate model
vqvae = VectorQuantizedVAE(CODE_DIM, CODE_SIZE).cuda()

# Define TB writer
writer = SummaryWriter(f'{LOGS_VQVAE_PATH}/{MODEL_NAME}')
writer.add_graph(vqvae, train_dataset[0].unsqueeze(0).cuda())
train_epochs(vqvae, train_dataloader, val_dataloader, \
             dict(epochs=500, lr=1e-4, grad_clip=1), writer, quiet=False)
writer.close()