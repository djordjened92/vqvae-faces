import functools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@functools.lru_cache(maxsize=32)
def image_positional_encoding(shape):
    """Generates *per-channel* positional encodings for 2d images.
    The positional encoding is a Tensor of shape (N, 2*C, H, W) of (x, y) pixel
    coordinates scaled to be between -.5 and .5.
    Args:
        shape: NCHW shape of image for which to generate positional encodings.
    Returns:
        The positional encodings.
    """
    n, c, h, w = shape
    zeros = torch.zeros(n, c, h, w)
    return torch.cat(
        (
            (torch.arange(-0.5, 0.5, 1 / h)[None, None, :, None] + zeros),
            (torch.arange(-0.5, 0.5, 1 / w)[None, None, None, :] + zeros),
        ),
        dim=1,
    )


@functools.lru_cache(maxsize=32)
def _get_causal_mask(size, mask_center):
    """Generates causal masks for attention weights."""
    return torch.tril(torch.ones((size, size)), diagonal=-int(mask_center))

class CausalAttention(nn.Module):
    """Autoregresively masked, multihead self-attention layer.
    Autoregressive masking means that the current pixel can only attend to itself,
    pixels to the left, and pixels above. When mask_center=True, the current pixel does
    not attent to itself.
    This Module generalizes attention to use 2D convolutions instead of fully connected
    layers. As such, the input is expected to be 4D image tensors.
    """

    def __init__(
        self,
        in_channels,
        n_heads=1,
        embed_channels=None,
        out_channels=None,
        mask_center=False,
        extra_input_channels=0,
    ):
        """Initializes a new CausalAttention instance.
        Args:
            in_channels: Number of input channels.
            n_heads: Number of causal self-attention heads.
            embed_channels: Number of embedding channels. Defaults to in_channels.
            out_channels: Number of output channels. Defaults to in_channels.
            extra_input_channels: Extra input channels which are only used to compute
                the embeddings and not the attention weights since doing so may break
                the autoregressive property. For example, in [1] these channels include
                the original input image.
            mask_center: Whether to mask the center pixel of the attention matrices.
        """
        super().__init__()
        self._n_heads = n_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_center = mask_center


        self._q = nn.Conv2d(
            in_channels=in_channels, out_channels=self._embed_channels, kernel_size=1
        )
        self._kv = nn.Conv2d(
            in_channels=in_channels + extra_input_channels,
            out_channels=self._embed_channels + self._out_channels,
            kernel_size=1,
        )
        
        self._proj = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=1,
        )

    def forward(self, x, extra_x=None):
        """Computes the forward pass.
        Args:
            x: The input used to compute both embeddings and attention weights.
            extra_x: Extra channels concatenated with 'x' only used to compute the
                embeddings. See the 'extra_input_channels' argument for more info.
        Returns:
            The result of the forward pass.
        """

        def _to_multihead(t):
            """Reshapes an (N, C, H, W) tensor into (N, n_heads, H * W, head_size)."""
            c = t.shape[1]
            t = t.view(n, self._n_heads, c // self._n_heads, -1)
            return t.transpose(2, 3)

        n, _, h, w = x.shape

        # Compute the query, key, and value.
        q = _to_multihead(self._q(x))
        if extra_x is not None:
            x = torch.cat((x, extra_x), dim=1)
        k, v = self._kv(x).split([self._embed_channels, self._out_channels], dim=1)
        k, v = _to_multihead(k), _to_multihead(v)

        # Compute the causual attention weights.
        mask = (
            _get_causal_mask(h * w, self._mask_center)
            .view(1, 1, h * w, h * w)
            .to(next(self.parameters()).device)
        )
        attn = (q @ k.transpose(2, 3)) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)
        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)

        # Attent to output for each head, stack, and project.
        out = (attn @ v).transpose(2, 3).contiguous().view(n, -1, h, w)
        return self._proj(out)