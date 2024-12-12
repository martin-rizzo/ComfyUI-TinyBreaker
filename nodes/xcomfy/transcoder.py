"""
File     : xconfy/transcoder.py
Purpose  : The ComfyUI object transmitted across the TRANSCODER lines.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : Nov 30, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import math
import torch
import torch.nn as nn
from ..utils.system                    import logger
from .objects                          import VAE
from torchvision.transforms.functional import gaussian_blur
from comfy                             import model_management


class Transcoder:
    """
    The ComfyUI object transmitted across the TRANSCODER lines.

    This object encapsulates the necessary components for transcoding latent
    representations between different models (e.g., from a SDXL to SD15).
    It allows for flexible configuration using either a custom model, or
    a pair of decoder and encoder VAEs

    **Important:** Do not create an instance of this class directly.
    Instead, use the class methods `from_model` or `from_decoder_encoder`.
    """
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


    @classmethod
    def from_model(cls,
                   model : nn.Module,
                   device: torch.device = None,
                   dtype : torch.dtype  = None,
                   ) -> "Transcoder":
        """
        Creates a new `Transcoder` instance using a custom model.

        This method is used when you want to use a single custom model for transcoding.
        The model is expected to handle the entire transcoding process internally.

        Args:
            model : The model to use for transcoding.
            device: The device where the model will be loaded.
                    If no specified, the default ComfyUI VAE device will be used.
            dtype : The data type used for the model.
                    If no specified, the default ComfyUI VAE data type will be used.
        Returns:
            A new `Transcoder` instance configured with the provided model.
        """
        working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if device is None:
            device = model_management.vae_device()
        if dtype is None:
            dtype = model_management.vae_dtype(device, working_dtypes)

        model.to(device=device, dtype=dtype)
        transcoder = cls(model=model, model_device=device, model_dtype=dtype)
        return transcoder


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
        It's important to note that blurring is not always necessary and may not be
        needed for all transcoding scenarios.

        Args:
            decoder (VAE): The VAE model used for decoding.
            encoder (VAE): The VAE model used for encoding.
            gaussian_blur_sigma (float, optional): The level of Gaussian blur to apply after decoding and before encoding.

       Returns:
            A new `Transcoder` instance configured with the provided decoder and encoder.
        """
        return cls(decoder=decoder, encoder=encoder, gaussian_blur_sigma=gaussian_blur_sigma)


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



