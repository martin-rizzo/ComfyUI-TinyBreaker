"""
File    : tiny_transcoder_model.py
Purpose : A lightweight model to convert images from one latent space to another.
          (based on the Tiny AutoEncoder model by @madebyollin)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 30, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import math
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

def gaussian_blur_auto_kernel(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Returns a new tensor which is a gaussian blur of the input tensor.

    This function is identical to `torchvision.transforms.functional.gaussian_blur()`
    except that it automatically calculates the kernel size based on the sigma value.

    Args:
        tensor (torch.Tensor): The input tensor with shape [..., H, W].
        sigma        (float) : The standard deviation of the gaussian blur.
    """
    if sigma <= 0.0:
        return tensor

    # calculate the kernel_size for the given sigma
    # the result should always be odd and equal to or greater than 3
    kernel_size = ((sigma - 0.8) / 0.3 + 1) * 2 + 1
    kernel_size = int( max(3.0, math.ceil(kernel_size)) )
    if kernel_size % 2 == 0:
        kernel_size += 1

    # apply gaussian blur using the calculated kernel_size and sigma
    return gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)


#------------------------------ MODEL BLOCKS -------------------------------#

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Conv3x3(nn.Conv2d):
   def __init__(self,
                input_channels: int,
                output_channels: int,
                **kwargs
                ):
        super().__init__(input_channels, output_channels, kernel_size=3, padding=1, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(self,
                 n_in : int,
                 n_out: int
                 ):
        super().__init__()
        self.conv = nn.Sequential(Conv3x3(n_in, n_out), nn.ReLU(), Conv3x3(n_out, n_out), nn.ReLU(), Conv3x3(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class Decoder(nn.Sequential):
    def __init__(self,
                 latent_channels      : int  =  4,
                 intermediate_channels: int  = 64,
                 convolutional_layers : int  =  3,
                 res_blocks_per_layer : int  =  3,
                 add_output_rgb_layer : bool = True
                 ):
        """
        The decoder part of the model.
        Args:
            latent_channels       (int): Number of channels in the input latent space.
            intermediate_channels (int): Number of channels in the intermediate layers.
            convolutional_layers  (int): Number of convolutional layers.
            res_blocks_per_layer  (int): Number of residual blocks per layer.
            add_output_rgb_layer (bool): If True, a last layer is added to convert the output to 3 channels.
        """
        ichans    = intermediate_channels
        rgb_chans = 3
        decoder   = []

        # decoder reference 64/3/3:
        #     Clamp(), Conv3x3(latent_channels, 64), nn.ReLU(),
        #     ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64), nn.Upsample(scale_factor=2), Conv3x3(64, 64, bias=False),
        #     ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64), nn.Upsample(scale_factor=2), Conv3x3(64, 64, bias=False),
        #     ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64), nn.Upsample(scale_factor=2), Conv3x3(64, 64, bias=False),
        #     ResidualBlock(64, 64), Conv3x3(64, rgb_chans),

        decoder.append( Clamp()                          )
        decoder.append( Conv3x3(latent_channels, ichans) )
        decoder.append( nn.ReLU()                        )

        for _ in range(convolutional_layers):
            for _ in range(res_blocks_per_layer):
                decoder.append( ResidualBlock(ichans, ichans)           )
            decoder.append(         nn.Upsample(scale_factor=2)         )
            decoder.append(         Conv3x3(ichans, ichans, bias=False) )

        if add_output_rgb_layer:
            decoder.append( ResidualBlock(ichans, ichans) )
            decoder.append( Conv3x3(64, rgb_chans)        )

        super().__init__(*decoder)


class Encoder(nn.Sequential):
    def __init__(self,
                 latent_channels      : int  =  4,
                 intermediate_channels: int  = 64,
                 convolutional_layers : int  =  3,
                 res_blocks_per_layer : int  =  3,
                 add_input_rgb_layer  : bool = True
                 ):
        """
        The encoder part of the model.
        Args:
            latent_channels           (int): Number of channels in the output latent space.
            intermediate_channels     (int): Number of channels in the intermediate layers.
            convolutional_layers      (int): Number of convolutional layers.
            residual_blocks_per_layer (int): Number of residual blocks per layer.
            add_input_rgb_layer      (bool): If True, a first layer is added to receive 3 channels as input.
        """
        ichans    = intermediate_channels
        rgb_chans = 3
        encoder   = []

        # encoder reference 64/3/3:
        #     Conv3x3(rgb_chans, 64), ResidualBlock(64, 64),
        #     Conv3x3(64, 64, stride=2, bias=False), ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64),
        #     Conv3x3(64, 64, stride=2, bias=False), ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64),
        #     Conv3x3(64, 64, stride=2, bias=False), ResidualBlock(64, 64), ResidualBlock(64, 64), ResidualBlock(64, 64),
        #     Conv3x3(64, latent_channels),

        if add_input_rgb_layer:
            encoder.append( Conv3x3(rgb_chans, ichans)    )
            encoder.append( ResidualBlock(ichans, ichans) )

        for _ in range(convolutional_layers):
            encoder.append( Conv3x3(ichans, ichans, stride=2, bias=False) )
            for _ in range(res_blocks_per_layer):
                encoder.append( ResidualBlock(ichans, ichans) )

        encoder.append( Conv3x3(ichans, latent_channels) )

        super().__init__(*encoder)


#========================== TINY TRANSCODER MODEL ==========================#
class TinyTranscoderModel(nn.Module):

    def __init__(self,
                 input_channels               : int =  4,
                 output_channels              : int =  4,
                 decoder_intermediate_channels: int = 64,
                 decoder_convolutional_layers : int =  3,
                 decoder_residual_blocks      : int =  3,
                 encoder_intermediate_channels: int = 64,
                 encoder_convolutional_layers : int =  3,
                 encoder_res_blocks_per_layer : int =  3,
                 use_internal_rgb_layer       : bool = True,
                 use_gaussian_blur_layer      : bool = True
                 ):
        super().__init__()

        self.decoder = Decoder(input_channels,
                               decoder_intermediate_channels,
                               decoder_convolutional_layers,
                               decoder_residual_blocks,
                               add_output_rgb_layer = use_internal_rgb_layer
                               )
        self.encoder = Encoder(output_channels,
                               encoder_intermediate_channels,
                               encoder_convolutional_layers,
                               encoder_res_blocks_per_layer,
                               add_input_rgb_layer = use_internal_rgb_layer
                               )

        # built-in normalization parameters for the input latent image
        # (these values should not be applied to the input latent as they ware learned during training)
        self.built_in_input_scale = torch.nn.Parameter(torch.tensor([0.13025], dtype=torch.float16))
        self.built_in_input_shift = torch.nn.Parameter(torch.tensor([0.0    ], dtype=torch.float16))

        # built-in normalization parameters for the output latent image
        # (these values should not be applied to the output latent as they ware learned during training)
        self.built_in_output_scale = torch.nn.Parameter(torch.tensor([0.18215], dtype=torch.float16))
        self.built_in_output_shift = torch.nn.Parameter(torch.tensor([0.0    ], dtype=torch.float16))

        # sigma for gaussian blur between decoder and encoder (usually 0.0)
        # (this value is a kind of hack to allow softening the error between encoder and decoder)
        if use_gaussian_blur_layer:
            self.gaussian_blur_sigma = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float16))
        else:
            self.gaussian_blur_sigma = None

        # make sure that some parameters are immutable (i.e. not trainable)
        self.immutable_params = ["built_in_input_scale", "built_in_input_shift",
                                 "built_in_output_scale", "built_in_output_shift",
                                 "gaussian_blur_sigma"]
        for name in self.immutable_params:
            param = getattr(self, name, None)
            if param is not None:
                param.requires_grad = False


    @classmethod
    def infer_model_config(cls, state_dict: dict) -> dict:
        # TODO: implement this method to infer the model configuration from a state dictionary
        config = {
            "input_channels"                :   4 ,
            "output_channels"               :   4 ,
            "decoder_intermediate_channels" :  64 ,
            "decoder_convolutional_layers"  :   3 ,
            "decoder_residual_blocks"       :   3 ,
            "encoder_intermediate_channels" :  64 ,
            "encoder_convolutional_layers"  :   3 ,
            "encoder_res_blocks_per_layer"  :   3 ,
            "use_internal_rgb_layer"        : True,
            "use_gaussian_blur_layer"       : True,
        }
        return config

    @classmethod
    def from_state_dict(cls, state_dict: dict, prefix: str = "") -> "TinyTranscoderModel":

        # TODO: implement automatic detection of the model configuration from the state dictionary
        if prefix == "?":
            prefix = "" # detect_prefix(state_dict, "decoder.10.conv.4.bias", "encoder.12.conv.2.weight")

        transcoder = cls()

        # filter the state dictionary if a prefix is provided
        # (this allows loading of partial models or sub-models from a larger state dictionary)
        if prefix:
            prefix_length = len(prefix)
            state_dict = {key[prefix_length:]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}

        transcoder.load_state_dict(state_dict, strict=False, assign=False)
        return transcoder


    @property
    def dtype(self):
        """Returns the data type of the model parameters."""
        return next(self.decoder.parameters()).dtype


    @property
    def device(self):
        """Returns the device on which the model parameters are located."""
        return next(self.decoder.parameters()).device


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, output_channels, height, width).
        """
        if not self.training:
            x = x.to(self.device, dtype=self.dtype)
        x = self.decoder(x).clamp(0, 1)
        if self.gaussian_blur_sigma:
            x = gaussian_blur_auto_kernel(x, sigma=self.gaussian_blur_sigma.sum().item())
        x = self.encoder(x)
        return x


    @torch.no_grad()
    def compatible_transcode(self, x):
        """Forward pass through the transcoder model with ComfyUI/Auto1111 compatibility."""
        x = x.to(self.device, dtype=self.dtype)
        x = x - self.built_in_input_shift
        x = x * self.built_in_input_scale
        x = self(x)
        x = x / self.built_in_output_scale
        x = x + self.built_in_output_shift
        return x


    def freeze(self) -> None:
        """Freeze all parameters of the model to prevent them from being updated during inference."""
        for name, param in self.named_parameters():
            if name in self.immutable_params:
                continue
            param.requires_grad = False
        self.eval()


    def unfreeze(self) -> None:
        """Unfreeze all parameters of the model to allow them to be updated during training."""
        for name, param in self.named_parameters():
            if name in self.immutable_params:
                continue
            param.requires_grad = True
        self.train()

