"""
File    : tiny_transcoder_model.py
Purpose : A lightweight model to convert images from one latent space to another.
          It is designed to be used in any project with minimal external dependencies.
          Based on the Tiny AutoEncoder model by @madebyollin.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 30, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _GaussianBlur(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        # calculate the kernel_size for the given sigma
        # the result should always be odd and equal to or greater than 3
        unrounded_kernel_size = ((sigma - 0.8) / 0.3 + 1) * 2 + 1
        kernel_size           = int(unrounded_kernel_size)
        if kernel_size < unrounded_kernel_size:
            kernel_size += 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        self.sigma       = sigma
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.sigma <= 0.0:
            return x
        return gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)


class _Conv3x3(nn.Conv2d):
   def __init__(self,
                input_channels: int,
                output_channels: int,
                **kwargs
                ):
        super().__init__(input_channels, output_channels, kernel_size=3, padding=1, **kwargs)


class _ResidualBlock(nn.Module):
    def __init__(self,
                 n_in : int,
                 n_out: int
                 ):
        super().__init__()
        self.conv = nn.Sequential(_Conv3x3(n_in, n_out), nn.ReLU(), _Conv3x3(n_out, n_out), nn.ReLU(), _Conv3x3(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


#---------------------------------------------------------------------------#
class Decoder(nn.Sequential):
    """
    The decoder part of the model.
    Args:
        latent_channels       (int): Number of channels in the input latent space.
        intermediate_channels (int): Number of channels in the intermediate layers.
        convolutional_layers  (int): Number of convolutional layers.
        res_blocks_per_layer  (int): Number of residual blocks per layer.
        add_output_rgb_layer (bool): If True, a last layer is added to convert the output to 3 channels.
    """
    def __init__(self,
                 latent_channels      : int  =  4,
                 intermediate_channels: int  = 64,
                 convolutional_layers : int  =  3,
                 res_blocks_per_layer : int  =  3,
                 add_output_rgb_layer : bool = True
                 ):
        ichans    = intermediate_channels
        rgb_chans = 3
        decoder   = []

        decoder.append( _Clamp()                          )
        decoder.append( _Conv3x3(latent_channels, ichans) )
        decoder.append( nn.ReLU()                         )

        for _ in range(convolutional_layers):
            for _ in range(res_blocks_per_layer):
                decoder.append( _ResidualBlock(ichans, ichans)           )
            decoder.append(         nn.Upsample(scale_factor=2)          )
            decoder.append(         _Conv3x3(ichans, ichans, bias=False) )

        if add_output_rgb_layer:
            decoder.append( _ResidualBlock(ichans, ichans) )
            decoder.append( _Conv3x3(64, rgb_chans)        )

        super().__init__(*decoder)


#---------------------------------------------------------------------------#
class Encoder(nn.Sequential):
    """
    The encoder part of the model.
    Args:
        latent_channels           (int): Number of channels in the output latent space.
        intermediate_channels     (int): Number of channels in the intermediate layers.
        convolutional_layers      (int): Number of convolutional layers.
        residual_blocks_per_layer (int): Number of residual blocks per layer.
        add_input_rgb_layer      (bool): If True, a first layer is added to receive 3 channels as input.
    """
    def __init__(self,
                 latent_channels      : int  =  4,
                 intermediate_channels: int  = 64,
                 convolutional_layers : int  =  3,
                 res_blocks_per_layer : int  =  3,
                 add_input_rgb_layer  : bool = True
                 ):
        ichans    = intermediate_channels
        rgb_chans = 3
        encoder   = []

        if add_input_rgb_layer:
            encoder.append( _Conv3x3(rgb_chans, ichans)    )
            encoder.append( _ResidualBlock(ichans, ichans) )

        for _ in range(convolutional_layers):
            encoder.append( _Conv3x3(ichans, ichans, stride=2, bias=False) )
            for _ in range(res_blocks_per_layer):
                encoder.append( _ResidualBlock(ichans, ichans) )

        encoder.append( _Conv3x3(ichans, latent_channels) )

        super().__init__(*encoder)


#---------------------------------------------------------------------------#
class UnnormalizedLatentAdapter(nn.Module):
    """
    Adapts an unnormalized latent tensor to/from the expected format.

    Important: This transformation is designed for latents that are *not* normalized
    to have a mean of 0 and a standard deviation of 1. If the input latents are
    already normalized, this adaptation should not be used.

    Args:
        latent_format (str): The format of the latent to normalize. Must be one of the
                             following: "unknown", "raw", "sd15", "sdxl", "sd3", "flux"
        is_input     (bool): If True, applies the transformation to bring the latent to
                             the desired range (e.g., for use as input to a UNet model).
                             If False, applies the inverse transformation to return to
                             the original latent space (e.g., for vae decoding).
    """
    def __init__(self,
                 latent_format: str  = "unknown",
                 is_input     : bool = False,
                 ):
        super().__init__()
        SCALE_SHIFT_BY_LATENT_FORMAT = {
            "unknown": (1.     ,  0.    ), # <- used when the parameters are read from file
            "raw"    : (1.     ,  0.    ), # <- used when no scaling or shifting is needed
            "sd15"   : (0.18215,  0.    ), # parameters for Stable Diffusion v1.5 latents
            "sdxl"   : (0.13025,  0.    ), # parameters for Stable Diffusion XL latents
            "sd3"    : (1.5305 , -0.0609), # parameters for Stable Diffusion 3 latents
            "flux"   : (0.3611 , -0.1159), # parameters for Flux latents
        }
        scale_shift = SCALE_SHIFT_BY_LATENT_FORMAT.get(latent_format.lower(), (1.,0.))
        self.scale_factor = torch.nn.Parameter(torch.tensor(scale_shift[0], dtype=torch.float32), requires_grad=False)
        self.shift_factor = torch.nn.Parameter(torch.tensor(scale_shift[1], dtype=torch.float32), requires_grad=False)
        self.is_input     = is_input

    def forward(self, x):
        if self.is_input:
            return (x + self.shift_factor) * self.scale_factor
        else:
            return (x / self.scale_factor) - self.shift_factor

    def freeze(self) -> None:
        """Freeze all parameters of this module."""
        for param in self.parameters():
            param.requires_grad = False


#===========================================================================#
#////////////////////////// TINY TRANSCODER MODEL //////////////////////////#
#===========================================================================#

class TinyTranscoderModel(nn.Module):
    """
    This model transcodes latent images between different formats (e.g., SDXL to SD1.5).

    Args:
        input_channels               : Number of channels in the input latent image.
        output_channels              : Number of channels in the output latent image.
        decoder_intermediate_channels: Number of channels in the intermediate layers of the decoder.
        decoder_convolutional_layers : Number of convolutional layers in the decoder.
        decoder_res_blocks_per_layer : Number of residual blocks per layer in the decoder.
        encoder_intermediate_channels: Number of channels in the intermediate layers of the encoder.
        encoder_convolutional_layers : Number of convolutional layers in the encoder.
        encoder_res_blocks_per_layer : Number of residual blocks per layer in the encoder.
        use_internal_rgb_layer       : If True, uses internal layers to convert to/from RGB space.
        use_gaussian_blur_bridge     : If True, a Gaussian blur layer is added between the decoder and encoder.
        input_latent_format          : The format of the input latent images, must be one of the
                                       following: "unknown", "raw", "sd15", "sdxl", "sd3", "flux".
        output_latent_format         : The format of the output latent images, must be one of the
                                       following: "unknown", "raw", "sd15", "sdxl", "sd3", "flux".
    """
    def __init__(self,
                 input_channels               : int  =    4 ,
                 output_channels              : int  =    4 ,
                 decoder_intermediate_channels: int  =   64 ,
                 decoder_convolutional_layers : int  =    3 ,
                 decoder_res_blocks_per_layer : int  =    3 ,
                 encoder_intermediate_channels: int  =   64 ,
                 encoder_convolutional_layers : int  =    3 ,
                 encoder_res_blocks_per_layer : int  =    3 ,
                 use_internal_rgb_layer       : bool =  True,
                 use_gaussian_blur_bridge     : bool =  True,
                 input_latent_format          : str  = "unknown",
                 output_latent_format         : str  = "unknown",
                 ):
        super().__init__()

        # decoder layers
        self.decoder = Decoder(input_channels,
                               decoder_intermediate_channels,
                               decoder_convolutional_layers,
                               decoder_res_blocks_per_layer,
                               add_output_rgb_layer = use_internal_rgb_layer
                               )

        # optional gaussian blur layer
        # used as a bridge between decoder and encoder to soften errors
        if use_gaussian_blur_bridge:
            self.bridge = _GaussianBlur(0.5)
        else:
            self.bridge = nn.Identity()

        # encoder layers
        self.encoder = Encoder(output_channels,
                               encoder_intermediate_channels,
                               encoder_convolutional_layers,
                               encoder_res_blocks_per_layer,
                               add_input_rgb_layer = use_internal_rgb_layer
                               )

        # non-trainable adapters that handle normalization/denormalization
        # for compatibility with raw, unnormalized latent formats
        # (these are a kind of hacky workaround)
        self.unnormalized_adapter_in  = UnnormalizedLatentAdapter(input_latent_format , is_input=True )
        self.unnormalized_adapter_out = UnnormalizedLatentAdapter(output_latent_format, is_input=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # REF:
        #  x = [batch_size, input_channels, height, width]
        x = self.unnormalized_adapter_in(x)
        x = self.decoder(x).clamp(0, 1)
        x = self.bridge(x)
        x = self.encoder(x)
        x = self.unnormalized_adapter_out(x)
        return x


    def load_state_dict(self, state_dict, *args, **kwargs):
        """Overridden method to load model state keeping the adapter parameters frozen."""
        super().load_state_dict(state_dict, *args, **kwargs)
        self.unnormalized_adapter_in.freeze()
        self.unnormalized_adapter_out.freeze()
