def test_reconstruction(vqvae, batch_cnt = 2):
    test_iterator = iter(test_dataloader)
    x = [next(test_iterator).cuda() for i in range(batch_cnt)]
    x = torch.cat(x, 0)
    with torch.no_grad():
        z = vqvae.encode_code(x)
        x_recon = vqvae.decode_code(z)
    reconstructions = torch.cat((x, x_recon), axis=0)
    return reconstructions