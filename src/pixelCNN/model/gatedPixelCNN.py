import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from .pixelCNN import LayerNorm, MaskConv2d

class StackLayerNorm(nn.Module):
  def __init__(self, n_filters):
    super().__init__()
    self.h_layer_norm = LayerNorm(n_filters)
    self.v_layer_norm = LayerNorm(n_filters)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)
    vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
    return torch.cat((vx, hx), dim=1)

class GatedConv2d(nn.Module):
  def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
    super().__init__()

    self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                              padding=padding, bias=False)
    self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                padding=(0, padding), bias=False)
    self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, 
                            bias=False)
    self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1, 
                            bias=False)


    self.register_buffer('vmask', self.vertical.weight.data.clone())
    self.register_buffer('hmask', self.horizontal.weight.data.clone())

    self.vmask.fill_(1)
    self.hmask.fill_(1)

    # zero the bottom half rows of the vmask
    # No need for special color condition masking here since we get to see everything
    self.vmask[:, :, k // 2 + 1:, :] = 0

    # zero the right half of the hmask
    self.hmask[:, :, :, k // 2 + 1:] = 0
    if mask_type == 'A':
      self.hmask[:, :, :, k // 2] = 0
  
  def down_shift(self, x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)

  def forward(self, x):
    vx, hx = x.chunk(2, dim=1)

    self.vertical.weight.data *= self.vmask
    self.horizontal.weight.data *= self.hmask

    vx = self.vertical(vx)
    hx_new = self.horizontal(hx)
    # Allow horizontal stack to see information from vertical stack
    hx_new = hx_new + self.vtoh(self.down_shift(vx))

    # Gates
    vx_1, vx_2 = vx.chunk(2, dim=1)
    vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

    hx_1, hx_2 = hx_new.chunk(2, dim=1)
    hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
    hx_new = self.htoh(hx_new)
    hx = hx + hx_new

    return torch.cat((vx, hx), dim=1)

# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(nn.Module):
  def __init__(self, input_shape, size, n_layers=7):
    super().__init__()
    self.input_shape = input_shape
    self.size = size

    self.in_conv = MaskConv2d('A', 1, size, 7, padding=3)
    model = []
    for _ in range(n_layers - 2):
      model.extend([nn.ReLU(), StackLayerNorm(size), GatedConv2d('B', size, size, 7, padding=3)])
    model.extend([nn.ReLU(), StackLayerNorm(size)])
    self.out_conv = MaskConv2d('B', size, size, 7, padding=3)
    self.net = nn.Sequential(*model)

  def forward(self, x):
    out = (x.float() / (self.size - 1) - 0.5) / 0.5
    out = self.in_conv(out)
    out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
    out = self.out_conv(out)
    return out
  
  def loss(self, x):
    return OrderedDict(loss=F.cross_entropy(self(x), x.long()))

  def sample(self, n):
    samples = torch.zeros(n, *self.input_shape).long().cuda()
    with torch.no_grad():
      for r in range(self.input_shape[0]):
        for c in range(self.input_shape[1]):
            logits = self(samples)[:, :, r, c]
            probs = F.softmax(logits, dim=1)
            samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
    return samples