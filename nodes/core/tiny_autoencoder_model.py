"""
File    : tiny_autoencoder_model.py
Purpose : Custom implementation of the "Tiny Autoencoder Model" supporting independent
          encoder and decoder submodels.  The code has minimal dependencies and can
          be easily integrated into any project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 2, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as nn


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _Conv3x3(nn.Conv2d):
   def __init__(self,
                input_channels : int,
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
class Encoder(nn.Sequential):
    """
    The encoder part of the model.
    Args:
        input_channels       (int): Number of channels in the input image (usually 3 for RGB).
        output_channels      (int): Number of channels in the output latent space.
        hidden_channels      (int): Number of channels in the intermediate layers.
        convolutional_layers (int): Number of convolutional layers.
        res_blocks_per_layer (int): Number of residual blocks per layer.
    """
    def __init__(self,
                 input_channels       : int  =  3,
                 output_channels      : int  =  4,
                 hidden_channels      : int  = 64,
                 convolutional_layers : int  =  3,
                 res_blocks_per_layer : int  =  3,
                 ):
        channels = hidden_channels
        encoder  = []

        encoder.append( _Conv3x3(input_channels, channels) )
        encoder.append( _ResidualBlock(channels, channels) )

        for _ in range(convolutional_layers):
            encoder.append( _Conv3x3(channels, channels, stride=2, bias=False) )
            for _ in range(res_blocks_per_layer):
                encoder.append( _ResidualBlock(channels, channels) )

        encoder.append( _Conv3x3(channels, output_channels) )

        super().__init__(*encoder)


#---------------------------------------------------------------------------#
class Decoder(nn.Sequential):
    """
    The decoder part of the model.
    Args:
        input_channels        (int): Number of channels in the input latent space.
        output_channels       (int): Number of channels in the output image (usually 3 for RGB).
        hidden_channels       (int): Number of channels in the intermediate layers.
        convolutional_layers  (int): Number of convolutional layers.
        res_blocks_per_layer  (int): Number of residual blocks per layer.
    """
    def __init__(self,
                 input_channels      : int  =  4,
                 output_channels     : int  =  3,
                 hidden_channels     : int  = 64,
                 convolutional_layers: int  =  3,
                 res_blocks_per_layer: int  =  3,
                 ):
        channels = hidden_channels
        decoder  = []

        decoder.append( _Clamp()                           )
        decoder.append( _Conv3x3(input_channels, channels) )
        decoder.append( nn.ReLU()                          )

        for _ in range(convolutional_layers):
            for _ in range(res_blocks_per_layer):
                decoder.append( _ResidualBlock(channels, channels)           )
            decoder.append(         nn.Upsample(scale_factor=2)          )
            decoder.append(         _Conv3x3(channels, channels, bias=False) )

        decoder.append( _ResidualBlock(channels, channels) )
        decoder.append( _Conv3x3(64, output_channels)        )

        super().__init__(*decoder)


#===========================================================================#
#///////////////////////// TINY AUTOENCODER MODEL //////////////////////////#
#===========================================================================#

class TinyAutoencoderModel(nn.Module):

    def __init__(self, *,
                 image_channels               : int  =    3 ,
                 latent_channels              : int  =    4 ,
                 encoder_hidden_channels      : int  =   64 ,
                 encoder_convolutional_layers : int  =    3 ,
                 encoder_res_blocks_per_layer : int  =    3 ,
                 decoder_hidden_channels      : int  =   64 ,
                 decoder_convolutional_layers : int  =    3 ,
                 decoder_res_blocks_per_layer : int  =    3 ,
                 encoder_latent_format        : str  = "unknown",
                 decoder_latent_format        : str  = "unknown",
                 ):
        super().__init__()
        SCALE_SHIFT_BY_LATENT_FORMAT = {
            "unknown": (1.     ,  0.    ), # <- used when the scale/shift values will be read from the model's weights
            "sd15"   : (0.18215,  0.    ), # parameters for Stable Diffusion v1.5 latents
            "sdxl"   : (0.13025,  0.    ), # parameters for Stable Diffusion XL latents
            "sd3"    : (1.5305 , -0.0609), # parameters for Stable Diffusion 3 latents
            "flux"   : (0.3611 , -0.1159), # parameters for Flux latents
            }

        self.encoder = None
        self.decoder = None

        # configure the encoder submodel
        if encoder_hidden_channels and encoder_convolutional_layers and encoder_res_blocks_per_layer:
            self.encoder = Encoder(image_channels,
                                   latent_channels,
                                   encoder_hidden_channels,
                                   encoder_convolutional_layers,
                                   encoder_res_blocks_per_layer,
                                   )

        # configure the decoder submodel
        if decoder_hidden_channels and decoder_convolutional_layers and decoder_res_blocks_per_layer:
            self.decoder = Decoder(latent_channels,
                                   image_channels,
                                   decoder_hidden_channels,
                                   decoder_convolutional_layers,
                                   decoder_res_blocks_per_layer,
                                   )

        # these values are used to scale/shift the latent space when simulating a standard autoencoder
        self.encoder_scale_factor = SCALE_SHIFT_BY_LATENT_FORMAT[encoder_latent_format][0]
        self.encoder_shift_factor = SCALE_SHIFT_BY_LATENT_FORMAT[encoder_latent_format][1]
        self.decoder_shift_factor = SCALE_SHIFT_BY_LATENT_FORMAT[decoder_latent_format][0]
        self.decoder_scale_factor = SCALE_SHIFT_BY_LATENT_FORMAT[decoder_latent_format][1]
        self.simulate_std_autoencoder = False


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # this implementation supports that the encoder is not present
        # and can simulate the range of values returned by a standard autoencoder
        if not self.encoder:
            return None
        x = self.encoder(x)
        if self.simulate_std_autoencoder:
            x = (x / self.encoder_scale_factor) - self.encoder_shift_factor
        return x


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # this implementation supports that the decoder is not present
        # and can simulate the range of values returned by a standard autoencoder
        if not self.decoder:
            return None
        if self.simulate_std_autoencoder:
            x = (x + self.decoder_shift_factor) * self.decoder_scale_factor
        x = self.decoder(x)
        return x


    def load_state_dict(self, state_dict, strict = True, assign = False):
        state_dict = state_dict.copy()

        if "encoder.vae_scale" in state_dict:
            self.encoder_scale_factor = state_dict.pop("encoder.vae_scale").item()
        if "encoder.vae_shift" in state_dict:
            self.encoder_shift_factor = state_dict.pop("encoder.vae_shift").item()
        if "decoder.vae_scale" in state_dict:
            self.decoder_scale_factor = state_dict.pop("decoder.vae_scale").item()
        if "decoder.vae_shift" in state_dict:
            self.decoder_shift_factor = state_dict.pop("decoder.vae_shift").item()

        return super().load_state_dict(state_dict, strict, assign)

