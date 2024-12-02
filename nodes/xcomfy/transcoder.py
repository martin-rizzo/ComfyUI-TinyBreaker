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
from ..utils.system                    import logger
from .objects                          import VAE
from torchvision.transforms.functional import gaussian_blur


class Transcoder:

    def __init__(self,
                 model  : torch.nn.Module,
                 decoder: VAE,
                 encoder: VAE,
                 gaussian_blur_sigma: float
                 ):
        self.model               = model
        self.decoder             = decoder
        self.encoder             = encoder
        self.gaussian_blur_sigma = gaussian_blur_sigma


    @classmethod
    def from_decoder_encoder(cls,
                             decoder: VAE,
                             encoder: VAE,
                             gaussian_blur_sigma: float = 0.0
                             ) -> "Transcoder":
        """Creates an instance of Transcoder from a given decoder and encoder.
        Args:
            decoder (VAE): The VAE model used for decoding.
            encoder (VAE): The VAE model used for encoding.
            gaussian_blur_sigma (float): The level of Gaussian blur to apply after decoding and before encoding.
        """
        return cls(model=None, decoder=decoder, encoder=encoder, gaussian_blur_sigma=gaussian_blur_sigma)


    @classmethod
    def from_model(cls,
                   model: torch.nn.Module
                   ) -> "Transcoder":
        """Creates an instance of Transcoder from a given model.
        Args:
            model (torch.nn.Module): The model to use for transcoding.
        """
        return cls(model=model, decoder=None, encoder=None, gaussian_blur_sigma=0.0)


    def transcode(self, samples):
        """Transcode the input samples."""
        if self.model is None and (self.encoder is None or self.decoder is None):
            logger.debug("No transcoder model or encoder/decoder provided, input samples will not be transcoded.")
            return samples

        #-- using a transcoder model if available (preferable method) -------
        if self.model is not None:
            # if the model has a compatible transcode method, use it
            if hasattr(self.model, 'compatible_transcode'):
                return self.model.compatible_transcode(samples)
            else:
                return self.model(samples)

        #-- fallback to using the decoder + encoder -------------------------
        images = self.decoder.decode(samples)
        images = self._fix_images_shape(images)

        # apply post-decoder effects to improve the quality of the transcoded images
        if self.gaussian_blur_sigma>0.0:
            images = images.permute(0, 3, 1, 2)
            images = self._gaussian_blur(images, sigma=self.gaussian_blur_sigma)
            images = images.permute(0, 2, 3, 1)

        samples = self.encoder.encode(images)
        return samples


    def __call__(self, samples):
        """Alias for transcode method."""
        return self.transcode(samples)


    @staticmethod
    def _fix_images_shape(images):
        """Ensures that the images have a shape of [batch_size, height, width, channels].
        """
        height, width, channels = images.shape[-3], images.shape[-2], images.shape[-1]

        # combine batches if there are 5 dimensions in the images tensor
        #   before: (num_batches, batch_size, height, width, channels)
        #   after :        (total_batch_size, height, width, channels)
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
        return gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)


