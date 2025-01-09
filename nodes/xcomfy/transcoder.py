"""
File    : xconfy/transcoder.py
Purpose : The standard TRANSCODER object transmitted through ComfyUI's node system.
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
from ..utils.system                  import logger
from ..core.tiny_transcoder_model_ex import TinyTranscoderModelEx
from .vae                            import VAE
from comfy                           import model_management
from torchvision.transforms.functional import gaussian_blur


#===========================================================================#
#/////////////////////////////// TRANSCODER ////////////////////////////////#
#===========================================================================#

class Transcoder:
    """This class represents any TRANSCODER object transmitted through ComfyUI's node system."""

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str = "",
                        device    : torch.device = None,
                        dtype     : torch.dtype  = None,
                        ) -> "Transcoder":
        """
        Creates a new `Transcoder` instance from a state dictionary of a TinyTranscoder model.
        Args:
            state_dict: The state dictionary of a TinyTranscoder model.
            prefix    : The prefix that indicates where the model is located within state_dict.
            device    : The device where the model will be loaded.
                        If no specified, the default ComfyUI VAE device will be used.
            dtype     : The data type used for the model.
                        If no specified, the default ComfyUI VAE data type will be used.
        """

        # try to create a TinyTranscoder model from state_dict
        tiny_transcoder_model = TinyTranscoderModelEx.from_state_dict(state_dict, prefix)
        if not tiny_transcoder_model:
            raise ValueError("Transcoder: unsupported model type")

        # determine device and data type where the model will be loaded
        # (use comfyui default VAE device and data type if not specified)
        working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if device is None:
            device = model_management.vae_device()
        if dtype is None:
            dtype = model_management.vae_dtype(device, working_dtypes)

        print("##>> device:", device)
        print("##>> dtype:", dtype)

        tiny_transcoder_model.to(device=device, dtype=dtype).freeze()
        return cls(model=tiny_transcoder_model, model_device=device, model_dtype=dtype)


    @classmethod
    def from_decoder_encoder(cls,
                             decoder: VAE,
                             encoder: VAE,
                             gaussian_blur_sigma: float = 0.0
                             ) -> "Transcoder":
        """
        Creates a new `Transcoder` instance using separate decoder and encoder VAE models.

        The transcoder will perform transcoding by first decoding a latent representation
        using the provided `decoder` VAE, and then encoding the resulting image with the
        provided `encoder` VAE.

        **Intended Use:** This method is specifically designed for use with standard
        ComfyUI VAE models. If you are working with custom models, it is recommended
        to use the `from_model` method instead.

        **Image Processing:** The transcoder optionally applies a Gaussian blur to the
        decoded image *before* re-encoding. This step can help to smooth out artifacts
        and improve the overall quality of the transcoded images in some cases.
        It's important to note that blurring is not always necessary.

        Args:
            decoder (VAE): A standard VAE model that will be used for decoding.
            encoder (VAE): A standard VAE model that will be used for encoding.
            gaussian_blur_sigma (float, optional): The level of Gaussian blur to apply after decoding and before encoding.

        """
        return cls(decoder=decoder, encoder=encoder, gaussian_blur_sigma=gaussian_blur_sigma)


    def __init__(self,
                 model              : nn.Module    = None,
                 model_device       : torch.device = None,
                 model_dtype        : torch.dtype  = None,
                 decoder            : VAE          = None,
                 encoder            : VAE          = None,
                 gaussian_blur_sigma: float        = 0.0 ,
                 ):
        self.model               = model
        self.model_device        = model_device
        self.model_dtype         = model_dtype
        self.decoder             = decoder
        self.encoder             = encoder
        self.gaussian_blur_sigma = gaussian_blur_sigma

        if self.model is None and (self.encoder is None or self.decoder is None):
            logger.debug("No transcoder model or encoder/decoder provided, input samples will not be transcoded.")


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Transcode the input samples."""

        # use a transcoder model if available (preferable method)
        if self.model is not None:
            x = x.to(device=self.model_device, dtype=self.model_dtype)
            return self.model(x)

        # use decoder + gaussian blur + encoder (fallback method)
        if (self.decoder is not None) and (self.encoder is not None):
            sigma = kwargs.get("gaussian_blur_sigma", self.gaussian_blur_sigma)
            x = self.decoder.decode(x)
            x = self._fix_images_shape(x)
            x = self._gaussian_blur(x, sigma=sigma)
            return self.encoder.encode(x)

        # if no transcoder model or encoder/decoder is available,
        # return the input samples as is.
        return x


    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Transcode the input samples."""
        return self.forward(x, **kwargs)


    @staticmethod
    def _fix_images_shape(images):
        """Ensures that the images have a shape of [batch_size, height, width, channels].
        """
        height, width, channels = images.shape[-3], images.shape[-2], images.shape[-1]

        # combine batches if there are 5 dimensions in the images tensor
        if len(images.shape) == 5:
            images = images.reshape(-1, height, width, channels)
        # remove alpha channels if there are more than 3 channels per pixel
        if channels > 3:
            images = images[:,:,:,:3]
        return images


    @staticmethod
    def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Returns a new tensor identical to `tensor` but with each element blurred using a Gaussian kernel.
        Args:
            tensor (torch.Tensor): The input tensor with shape [..., H, W].
            sigma        (float) : The standard deviation of the Gaussian.
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
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
        tensor = tensor.permute(0, 2, 3, 1)
        return tensor



