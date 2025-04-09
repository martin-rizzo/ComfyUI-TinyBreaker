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
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as torch_nn


def _Conv3x3(in_channels: int, out_channels: int, nn, **kwargs) -> torch_nn.Conv2d:
    # this is a shortcut to create convolutional layers with specific parameters #
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs)


class _Clamp(torch_nn.Module):
    # this layer clamps the input tensor to a range of -3 to +3 #
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _ResidualBlock(torch_nn.Module):
    """
    A residual block with 3 convolutional layers.
    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
        nn      (optional): The neural network module to use. Defaults to `torch.nn`.
                            This parameter allows for the injection of custom or
                            optimized implementations of "nn" modules.
    """
    def __init__(self,
                 in_channels : int,
                 out_channels: int,
                 nn = None
                 ):
        super().__init__()
        if nn is None:
            nn = torch_nn
        self.conv = nn.Sequential( _Conv3x3( in_channels, out_channels, nn = nn),
                                   nn.ReLU(),
                                   _Conv3x3(out_channels, out_channels, nn = nn),
                                   nn.ReLU(),
                                   _Conv3x3(out_channels, out_channels, nn = nn),
                                   )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


#---------------------------------------------------------------------------#
def Encoder(in_channels         : int  =  3,
            out_channels        : int  =  4,
            hidden_channels     : int  = 64,
            convolutional_layers: int  =  3,
            res_blocks_per_layer: int  =  3,
            nn = None,
            ) -> torch_nn.Sequential:
    """
    The encoder part of the model.
    Args:
        in_channels          (int): Number of channels in the input image (usually 3 for RGB).
        out_channels         (int): Number of channels in the output latent space.
        hidden_channels      (int): Number of channels in the intermediate layers.
        convolutional_layers (int): Number of convolutional layers.
        res_blocks_per_layer (int): Number of residual blocks per layer.
        nn              (optional): The neural network module to use. Defaults to `torch.nn`.
                                    This parameter allows for the injection of custom or
                                    optimized implementations of "nn" modules.
    """
    if nn is None:
        nn = torch_nn
    chans   = hidden_channels
    encoder = []

    encoder.append( _Conv3x3(in_channels, chans, nn = nn) ) # self[0]
    encoder.append( _ResidualBlock(chans, chans, nn = nn) ) # self[1]

    for _ in range(convolutional_layers):
        encoder.append( _Conv3x3(chans, chans, nn = nn, stride=2, bias=False) )
        for _ in range(res_blocks_per_layer):
            encoder.append( _ResidualBlock(chans, chans, nn = nn) )

    encoder.append( _Conv3x3(chans, out_channels, nn = nn) )
    return nn.Sequential(*encoder)


#---------------------------------------------------------------------------#
def Decoder(in_channels         : int  =  4,
            out_channels        : int  =  3,
            hidden_channels     : int  = 64,
            convolutional_layers: int  =  3,
            res_blocks_per_layer: int  =  3,
            nn = None,
            ) -> torch_nn.Sequential:
    """
    The decoder part of the model.
    Args:
        in_channels          (int): Number of channels in the input latent space.
        out_channels         (int): Number of channels in the output image (usually 3 for RGB).
        hidden_channels      (int): Number of channels in the intermediate layers.
        convolutional_layers (int): Number of convolutional layers.
        res_blocks_per_layer (int): Number of residual blocks per layer.
        nn              (optional): The neural network module to use. Defaults to `torch.nn`.
                                    This parameter allows for the injection of custom or
                                    optimized implementations of "nn" modules.
    """
    if nn is None:
        nn = torch_nn
    chans = hidden_channels
    decoder  = []

    decoder.append( _Clamp()                              ) # self[0]
    decoder.append( _Conv3x3(in_channels, chans, nn = nn) ) # self[1]
    decoder.append( nn.ReLU()                             ) # self[2]

    for _ in range(convolutional_layers):
        for _ in range(res_blocks_per_layer):
            decoder.append( _ResidualBlock(chans, chans, nn = nn)       )
        decoder.append(     nn.Upsample(scale_factor=2)                 )
        decoder.append(     _Conv3x3(chans, chans, nn = nn, bias=False) )

    decoder.append( _ResidualBlock(chans, chans, nn = nn) )
    decoder.append( _Conv3x3(64, out_channels, nn = nn)   )

    return nn.Sequential(*decoder)


#===========================================================================#
#///////////////////////// TINY AUTOENCODER MODEL //////////////////////////#
#===========================================================================#

class TinyAutoencoderModel(torch_nn.Module):

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
                 nn = None
                 ):
        super().__init__()
        if nn is None:
            nn = torch_nn

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
                                   nn = nn,
                                   )

        # configure the decoder submodel
        if decoder_hidden_channels and decoder_convolutional_layers and decoder_res_blocks_per_layer:
            self.decoder = Decoder(latent_channels,
                                   image_channels,
                                   decoder_hidden_channels,
                                   decoder_convolutional_layers,
                                   decoder_res_blocks_per_layer,
                                   nn = nn
                                   )

        # these values are used to scale/shift the latent space when emulating a standard autoencoder
        self.encoder_scale_factor = SCALE_SHIFT_BY_LATENT_FORMAT[encoder_latent_format][0]
        self.encoder_shift_factor = SCALE_SHIFT_BY_LATENT_FORMAT[encoder_latent_format][1]
        self.decoder_scale_factor = SCALE_SHIFT_BY_LATENT_FORMAT[decoder_latent_format][0]
        self.decoder_shift_factor = SCALE_SHIFT_BY_LATENT_FORMAT[decoder_latent_format][1]
        self.emulate_std_autoencoder = False


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # this implementation supports that the encoder is not present
        if not self.encoder:
            return None

        if self.emulate_std_autoencoder:
            # standard autoencoders recive a range [-1, 1] for image input
            # but tiny expect range [0, 1]
            x.add_(1).mul_(0.5)

        x = self.encoder(x)

        if self.emulate_std_autoencoder:
            # standard autoencoders have particular scale/shift values
            # but tiny always has scale_factor=1.0 and shift_factor=0.0
            # so we need to adjust the latent space after encoding it
            x = (x / self.encoder_scale_factor) - self.encoder_shift_factor

        return x


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # this implementation supports that the decoder is not present
        if not self.decoder:
            return None

        if self.emulate_std_autoencoder:
            # tiny always has scale_factor=1.0 and shift_factor=0.0
            # but standard autoencoders have particular scale/shift values
            # so we need to adjust the latent space before decoding it
            x = (x + self.decoder_shift_factor).mul_(self.decoder_scale_factor)

        x = self.decoder(x)

        if self.emulate_std_autoencoder:
            # tiny generate a range [0, 1] for output images
            # but standard autoencoders use range [-1, 1]
            x.mul_(2).sub_(1)

        return x


    def load_state_dict(self, state_dict, *args, **kwargs):
        """Override the default load_state_dict method to handle scale/shift values."""
        state_dict = state_dict.copy()

        # the scale/shift values are not part of the model's weights
        # so we need to remove them from the state dict before loading it

        # this section implements the conventional loading of scale and shift values.
        # where the encoder and decoder share a common latent space
        if "vae_scale" in state_dict:
            vae_scale = state_dict.pop("vae_scale").item()
            self.encoder_scale_factor = vae_scale
            self.decoder_scale_factor = vae_scale
        if "vae_shift" in state_dict:
            vae_shift = state_dict.pop("vae_shift").item()
            self.encoder_shift_factor = vae_shift
            self.decoder_shift_factor = vae_shift

        # this section implements a custom approach to loading scale and shift values,
        # which allows the encoder and decoder to have independent latent spaces and,
        # consequently, different scale and shift parameters
        if "encoder.vae_scale" in state_dict:
            self.encoder_scale_factor = state_dict.pop("encoder.vae_scale").item()
        if "encoder.vae_shift" in state_dict:
            self.encoder_shift_factor = state_dict.pop("encoder.vae_shift").item()
        if "decoder.vae_scale" in state_dict:
            self.decoder_scale_factor = state_dict.pop("decoder.vae_scale").item()
        if "decoder.vae_shift" in state_dict:
            self.decoder_shift_factor = state_dict.pop("decoder.vae_shift").item()

        return super().load_state_dict(state_dict, *args, **kwargs)


    def get_encoder_dtype(self) -> torch.dtype:
        """Get the dtype of the encoder's weights."""
        return self.encoder[0].weight.dtype if self.encoder else None

    def get_encoder_device(self) -> torch.device:
        """Get the device where the encoder is located."""
        return self.encoder[0].weight.device if self.encoder else None

    def get_decoder_dtype(self) -> torch.dtype:
        """Get the dtype of the decoder's weights."""
        return self.decoder[1].weight.dtype if self.decoder else None

    def get_decoder_device(self) -> torch.device:
        """Get the device where the decoder is located."""
        return self.decoder[1].weight.device if self.decoder else None
