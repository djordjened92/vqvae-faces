import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input, cond=None):
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1

class PixelCNNResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.ModuleList([
            LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(0.2),
            MaskConv2d('B', size, size, 3, padding=1),
            LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(0.2),
            MaskConv2d('B', size, size, 3, padding=1),
            LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(0.2),
            MaskConv2d('B', size, size, 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.block:
            out = layer(out)
        return x + out


class PixelCNN(nn.Module):
    def __init__(self, input_shape, size, n_layers=7):
        super().__init__()
        self.size = size
        self.input_shape = input_shape

        model = nn.ModuleList([MaskConv2d('A', 1, size, 5, padding=2)])
        for _ in range(n_layers - 1):
            model.append(PixelCNNResBlock(size))
        model.extend([LayerNorm(size), nn.ReLU(), MaskConv2d('B', size, size, 1)])
        self.net = model

    def forward(self, x):
        out = x.float()#(x.float() / (self.size - 1) - 0.5) / 0.5
        for layer in self.net:
            out = layer(out)
        return out

    def loss(self, x):
        return OrderedDict(loss=F.cross_entropy(self(x), x.squeeze()))

    def sample(self, n):
        samples = torch.zeros(n, *self.input_shape).long().cuda()
        with torch.no_grad():
            for r in range(self.input_shape[0]):
                for c in range(self.input_shape[1]):
                    logits = self(samples.unsqueeze(dim=1))[:, :, r, c]
                    logits = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(logits, 1).squeeze(-1)
        return samples