"""
File    : autoencoder_model.py
Purpose : Custom VAE implementation supporting independent encoder and decoder submodels.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 2, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GroupNorm32(nn.GroupNorm):
    """Normalizes the input tensor using a 32-group normalization."""
    def __init__(self, channels: int):
        super().__init__(num_groups=32, num_channels=channels, eps=1e-6)


class _ResidualBlock(nn.Module):
    """Residual block with group normalization and SiLU activation."""
    def __init__(self,
                 input_channels : int,
                 output_channels: int
                 ):
        super().__init__()
        self.norm1 = _GroupNorm32(input_channels)
        self.act1  = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.norm2 = _GroupNorm32(output_channels)
        self.act2  = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1)
        if input_channels != output_channels:
            self.nin_shortcut = nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return self.nin_shortcut(residual) + x


class _SelfAttention(nn.Module):
    """Self-attention layer with group normalization."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm     = _GroupNorm32(channels)
        self.q        = nn.Conv2d(channels, channels, 1)
        self.k        = nn.Conv2d(channels, channels, 1)
        self.v        = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale    = channels ** -0.5

    def forward(self, x: torch.Tensor):
        batch_size, channels, h, w = x.shape

        # store the input for residual connection later
        residual = x

        # project the normalized input to query, key, and value
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # reshape to [batch_size, heads, h * w, channels] for
        # efficient attention calculation. (num_heads is 1)
        q = q.view(batch_size, 1, channels, h * w).transpose(2,3)
        k = k.view(batch_size, 1, channels, h * w).transpose(2,3)
        v = v.view(batch_size, 1, channels, h * w).transpose(2,3)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # reshape back to [batch_size, channels, h, w]
        x = x.transpose(2,3).view(batch_size, channels, h, w)

        # project and add the residual connection
        return residual + self.proj_out(x)


class _UpSample(nn.Module):
    """Upsamples the input tensor by a factor of 2, effectively doubling its height and width."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # up-sample by a factor of 2 and then apply convolution
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class _DownSample(nn.Module):
    """Downsamples the input tensor by a factor of 2, effectively halving its height and width."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        # add padding of (0, 1, 0, 1) before applying convolution
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


#---------------------------------------------------------------------------#
class Encoder(nn.Module):
    """
    Encoder that transforms an input image into a latent representation.

    This encoder uses an architecture based on ResNet blocks and downsampling layers
    to reduce the spatial resolution of the input while increasing the channel depth.

    Args:
        input_channels       (int): Number of channels of the input image. Default: 3 (RGB).
        output_channels      (int): Number of channels of the output latent representation. Default: 8.
        hidden_channels      (int): Number of base channels for the intermediate layers. Default: 128.
        channel_multipliers (list): List of multipliers for the channels in each downsampling layer. Default: [1, 2, 4, 4].
        res_blocks_per_layer (int): Number of ResNet blocks in each downsampling layer. Default: 2.

    """
    def __init__(self, *,
                 input_channels      : int  =     3,
                 output_channels     : int  =     8,
                 hidden_channels     : int  =   128,
                 channel_multipliers : list = [1, 2, 4, 4],
                 res_blocks_per_layer: int  =     2,
                 ):
        super().__init__()
        assert channel_multipliers[0] == 1, 'The first element of `channel_multipliers` must be 1.'

        channels_by_layer  = [mul * hidden_channels for mul in channel_multipliers]
        number_of_layers   = len(channels_by_layer)
        end_layer          = number_of_layers - 1
        layer_0_channels   = channels_by_layer[0]
        end_layer_channels = channels_by_layer[-1]

        # initial convolution block that maps input channels to the layer-0 channels
        self.conv_in = nn.Conv2d(input_channels, layer_0_channels, 3, stride=1, padding=1)

        # each "down layer" halves spatial size while increasing channel depth,
        # helping to capture abstract features from the input image
        self.down = nn.ModuleList()
        for layer in range(number_of_layers):
            previous_layer_channels = channels_by_layer[ max(0,layer-1) ]
            layer_channels          = channels_by_layer[ layer ]

            # create a module with multiple ResNet Blocks
            resnet_blocks = nn.ModuleList()
            resnet_blocks.append( _ResidualBlock(previous_layer_channels, layer_channels) )
            for _ in range(res_blocks_per_layer-1):
                resnet_blocks.append( _ResidualBlock(layer_channels, layer_channels) )

            # the "down layer" consists of multiple ResNet Blocks and a down-sampling
            # but attention: the last layer does NOT have a down-sampling
            down_layer = nn.Module()
            down_layer.block      = resnet_blocks
            down_layer.downsample = _DownSample(layer_channels) if layer != end_layer else nn.Identity()
            self.down.append(down_layer)

        # final encoder block (mid) using ResNet with self attention
        self.mid = nn.Module()
        self.mid.block_1 = _ResidualBlock(end_layer_channels, end_layer_channels)
        self.mid.attn_1  = _SelfAttention(end_layer_channels)
        self.mid.block_2 = _ResidualBlock(end_layer_channels, end_layer_channels)

        # final convolution block that maps from the end-layer channels to output channels
        self.norm_out = _GroupNorm32(end_layer_channels)
        self.act_out  = nn.SiLU()
        self.conv_out = nn.Conv2d(end_layer_channels, output_channels, 3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                x = block(x)
            x = down.downsample(x)
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x


    @property
    def dtype(self):
        """Returns the data type of the encoder parameters."""
        return self.mid.block_1.conv1.weight.dtype


    @property
    def device(self):
        """Returns the device on which the encoder parameters are located."""
        return self.mid.block_1.conv1.weight.device


#---------------------------------------------------------------------------#
class Decoder(nn.Module):
    """
    """
    def __init__(self, *,
                 input_channels      : int  =     4,
                 output_channels     : int  =     3,
                 hidden_channels     : int  =   128,
                 channel_multipliers : list = [1, 2, 4, 4],
                 res_blocks_per_layer: int  =     2,
                 ):
        super().__init__()
        assert channel_multipliers[0] == 1, 'The first element of `channel_multipliers` must be 1.'

        channels_by_layer  = [mul * hidden_channels for mul in channel_multipliers]
        number_of_layers   = len(channels_by_layer)
        layer_0            = 0
        end_layer          = number_of_layers - 1
        layer_0_channels   = channels_by_layer[0]
        end_layer_channels = channels_by_layer[-1]

        # initial convolution block that maps from input channels to the end-layer channels
        self.conv_in = nn.Conv2d(input_channels, end_layer_channels, 3, stride=1, padding=1)

        # initial decoder block (mid) using ResNet with self attention
        self.mid = nn.Module()
        self.mid.block_1 = _ResidualBlock(end_layer_channels, end_layer_channels)
        self.mid.attn_1  = _SelfAttention(end_layer_channels)
        self.mid.block_2 = _ResidualBlock(end_layer_channels, end_layer_channels)

        # each "up layer" doubles the spatial size while decreasing channel depth,
        # reconstructing the image from the latent representation
        self.up = nn.ModuleList()
        for layer in reversed(range(number_of_layers)):
            previous_channels = channels_by_layer[ min(layer+1,end_layer) ]
            layer_channels    = channels_by_layer[ layer ]

            # create a module with multiple ResNet Blocks
            resnet_blocks = nn.ModuleList()
            resnet_blocks.append( _ResidualBlock(previous_channels, layer_channels) )
            for _ in range(res_blocks_per_layer):
                resnet_blocks.append( _ResidualBlock(layer_channels, layer_channels) )

            # the "up layer" consists of multiple ResNet blocks and a up-sampling
            # but attention: the first layer does NOT have a up-sampling
            up_layer = nn.Module()
            up_layer.block    = resnet_blocks
            up_layer.upsample = _UpSample(layer_channels) if layer != layer_0 else nn.Identity()
            self.up.insert(0, up_layer) # <- inserted in inversed order to be consistent with the checkpoint

        # final convolution block that maps from the layer-0 channels to output channels
        self.norm_out = _GroupNorm32(layer_0_channels)
        self.act_out  = nn.SiLU()
        self.conv_out = nn.Conv2d(layer_0_channels, output_channels, 3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        for up in reversed(self.up):
            for block in up.block:
                x = block(x)
            x = up.upsample(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x


    @property
    def dtype(self):
        """Returns the data type of the decoder parameters."""
        return self.mid.block_1.conv1.weight.dtype


    @property
    def device(self):
        """Returns the device on which the decoder parameters are located."""
        return self.mid.block_1.conv1.weight.device


#===========================================================================#
#//////////////////////////// AUTOENCODER MODEL ////////////////////////////#
#===========================================================================#

class AutoencoderModel(nn.Module):
    """
    Custom VAE implementation supporting independent encoder and decoder submodels.


    """
    def __init__(self, *,
                 image_channels              : int   =     3,
                 latent_channels             : int   =     4,
                 pre_quant_channels          : int   =     4,
                 encoder_hidden_channels     : int   =   128,
                 encoder_channel_multipliers : list  = [1, 2, 4, 4],
                 encoder_res_blocks_per_layer: int   =     2,
                 decoder_hidden_channels     : int   =   128,
                 decoder_channel_multipliers : list  = [1, 2, 4, 4],
                 decoder_res_blocks_per_level: int   =     2,
                 use_deterministic_encoding  : bool  =  True,
                 use_double_encoding_channels: bool  =  True,
                 ):
        super().__init__()
        self.use_deterministic_encoding   = use_deterministic_encoding
        self.use_double_encoding_channels = use_double_encoding_channels
        self.encoder = None
        self.decoder = None
        encoder_output_channels = pre_quant_channels
        encoder_latent_channels = latent_channels
        decoder_latent_channels = latent_channels
        decoder_input_channels  = pre_quant_channels

        # the double encoding channels is used to generate the mean and logvariance
        if use_double_encoding_channels:
            encoder_output_channels = 2 * pre_quant_channels
            encoder_latent_channels = 2 * latent_channels

        # configure the encoder submodel
        if encoder_hidden_channels and encoder_channel_multipliers and encoder_res_blocks_per_layer:
            self.encoder = Encoder(input_channels       = image_channels,
                                   output_channels      = encoder_output_channels,
                                   hidden_channels      = encoder_hidden_channels,
                                   channel_multipliers  = encoder_channel_multipliers,
                                   res_blocks_per_layer = encoder_res_blocks_per_layer)
            self.quant_conv = nn.Conv2d(encoder_output_channels, encoder_latent_channels, 1)

        # configure the decoder submodel
        if decoder_hidden_channels and decoder_channel_multipliers and decoder_res_blocks_per_level:
            self.post_quant_conv = nn.Conv2d(decoder_latent_channels, decoder_input_channels, 1)
            self.decoder = Decoder(input_channels       = decoder_input_channels,
                                output_channels      = image_channels,
                                hidden_channels      = decoder_hidden_channels,
                                channel_multipliers  = decoder_channel_multipliers,
                                res_blocks_per_layer = decoder_res_blocks_per_level)


    def encode(self, x):
        if not self.encoder:
            return None
        x = self.encoder(x)
        x = self.quant_conv(x)
        return self._sample_gaussian_distribution(x, self.use_double_encoding_channels, self.use_deterministic_encoding)


    def decode(self, x):
        if not self.decoder:
            return None
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x


    @staticmethod
    def _sample_gaussian_distribution(x, double_channels: bool, deterministic: bool, /):

        # if we are not using double encoding channels, just return the encoded image
        if not double_channels:
            return x

        # extract the mean and log variances from the encoded image
        # but if deterministic, just return the mean as latent image
        mean_values, log_variances = torch.chunk(x, 2, dim=1)
        if deterministic:
            return mean_values.clone()

        # generate the latent image taking a sample from the gaussian distribution
        standard_deviation_values = torch.exp(0.5 * torch.clamp(log_variances, -30.0, 20.0))
        return mean_values + standard_deviation_values * torch.randn_like(standard_deviation_values, device=x.device)

