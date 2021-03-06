import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from .causal_attention import CausalAttention, image_positional_encoding
from .base import AutoregressiveModel

def _elu_conv_elu(conv, x):
    return F.elu(conv(F.elu(x)))

class CausalConv2d(nn.Conv2d):
    """A Conv2d layer masked to respect the autoregressive property.
    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.
    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=False  [1 1 0],    mask_center=True  [1 0 0],
                           [0 0 0]]                      [0 0 0]
    In [1], they refer to the left masks as 'type A' and right as 'type B'.
    NOTE: This layer does *not* implement autoregressive channel masking.
    """

    def __init__(self, mask_center, *args, **kwargs):
        """Initializes a new CausalConv2d instance.
        Args:
            mask_center: Whether to mask the center pixel of the convolution filters.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.zeros((i, o, h, w))
        mask.data[:, :, : h // 2, :] = 1
        mask.data[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x.float())

class GatedActivation(nn.Module):
    """Gated activation function as introduced in [2].
    The function computes actiation_fn(f) * sigmoid(g). The f and g correspond to the
    top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self, activation_fn=torch.tanh):
        """Initializes a new GatedActivation instance.
        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        x, gate = x[:, : c // 2, :, :], x[:, c // 2 :, :, :]
        return self._activation_fn(x) * torch.sigmoid(gate)


class ResidualBlock(nn.Module):
    """Residual block with a gated activation function."""

    def __init__(self, n_channels):
        """Initializes a new ResidualBlock.

        Args:
            n_channels: The number of input and output channels.
        """
        super().__init__()
        self._input_conv = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=2, padding=1
        )
        self._output_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            kernel_size=2,
            padding=1,
        )
        self._activation = GatedActivation()

    def forward(self, x):
        _, c, h, w = x.shape
        out = _elu_conv_elu(self._input_conv, x)[:, :, :h, :w]
        out = self._activation(self._output_conv(out)[:, :, :h, :w])
        return x + out


class PixelSNAILBlock(nn.Module):
    """Block comprised of a number of residual blocks plus one attention block.

    Implements Figure 5 of [1].
    """

    def __init__(
        self,
        n_channels,
        input_img_channels=1,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
    ):
        """Initializes a new PixelSnailBlock instance.

        Args:
            n_channels: Number of input and output channels.
            input_img_channels: The number of channels in the original input_img. Used
                for the positional encoding channels and the extra channels for the key
                and value convolutions in the attention block.
            n_residual_blocks: Number of residual blocks.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
        """
        super().__init__()

        def conv(in_channels):
            return nn.Conv2d(in_channels, out_channels=n_channels, kernel_size=1)

        self._residual = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)]
        )
        self._attention = CausalAttention(
            in_channels=n_channels + 2 * input_img_channels,
            embed_channels=attention_key_channels,
            out_channels=attention_value_channels,
            mask_center=True,
            extra_input_channels=input_img_channels,
        )
        self._residual_out = conv(n_channels)
        self._attention_out = conv(attention_value_channels)
        self._out = conv(n_channels)

    def forward(self, x, input_img):
        """Computes the forward pass.

        Args:
            x: The input.
            input_img: The original image only used as input to the attention blocks.
        Returns:
            The result of the forward pass.
        """
        res = self._residual(x)
        pos = image_positional_encoding(input_img.shape).to(res.device)
        attn = self._attention(torch.cat((pos, res), dim=1), input_img)
        res, attn = (
            _elu_conv_elu(self._residual_out, res),
            _elu_conv_elu(self._attention_out, attn),
        )
        return _elu_conv_elu(self._out, res + attn)


class PixelSNAIL(AutoregressiveModel):
    """The PixelSNAIL model.

    Unlike [1], we implement skip connections from each block to the output.
    We find that this makes training a lot more stable and allows for much faster
    convergence.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_channels=64,
        n_pixel_snail_blocks=8,
        n_residual_blocks=2,
        attention_key_channels=4,
        attention_value_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelSNAIL instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output_channels.
            n_channels: Number of channels to use for convolutions.
            n_pixel_snail_blocks: Number of PixelSNAILBlocks.
            n_residual_blocks: Number of ResidualBlock to use in each PixelSnailBlock.
            attention_key_channels: Number of channels (dims) for the attention key.
            attention_value_channels: Number of channels (dims) for the attention value.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input = CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
        )
        self._pixel_snail_blocks = nn.ModuleList(
            [
                PixelSNAILBlock(
                    n_channels=n_channels,
                    input_img_channels=in_channels,
                    n_residual_blocks=n_residual_blocks,
                    attention_key_channels=attention_key_channels,
                    attention_value_channels=attention_value_channels,
                )
                for _ in range(n_pixel_snail_blocks)
            ]
        )
        self._output = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            ),
            nn.Conv2d(
                in_channels=n_channels // 2, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        # x = x.float()
        input_img = x
        x = self._input(x)
        for block in self._pixel_snail_blocks:
            x = x + block(x, input_img)
        return self._output(x)
    
    def loss(self, x):
        return OrderedDict(loss=F.cross_entropy(self(x), x.squeeze(1)))

    # def loss(self, x):
    #     preds = self(x)
    #     batch_size = x.shape[0]
    #     x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    #     loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
    #     return loss.sum(dim=1).mean()