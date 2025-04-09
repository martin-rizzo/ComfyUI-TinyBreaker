"""
File    : xcomfy/transcoder.py
Purpose : The standard TRANSCODER object transmitted through ComfyUI's node wires.
          This code is the bridge between custom Transcoder models and the node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 30, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import torch
from ..utils.system                    import logger
from ..core.models.transcoder_model_ex import TranscoderModelEx
from .vae                              import VAE
from comfy                             import model_management
from torchvision.transforms.functional import gaussian_blur


def _create_custom_transcoder_model(state_dict: dict,
                                    prefix    : str,
                                    filename  : str,
                                    ) -> tuple[object, list]:
    """
    Main function to create a custom Transcoder model from the given state_dict.
    Here must be added the code to detect and instantiate any custom Transcoder model.

    Args:
        state_dict: A dictionary containing the tensor parameters of the model.
        prefix    : A prefix indicating which of the tensors in state_dict belong to the model.
        filename  : The name of the file from which state_dict was loaded.
    Returns:
        A tuple containing the Transcoder model and a list of missing keys (if any)
    """
    # only one type of Transcoder is supported
    # therefore no autodetection of models is performed
    logger.info(f"Loading TranscoderModelEx from '{filename}'")
    transcoder_model, _, missing_keys, _ = \
        TranscoderModelEx.from_state_dict(state_dict, prefix)

    return transcoder_model, missing_keys


#===========================================================================#
#/////////////////////////////// TRANSCODER ////////////////////////////////#
#===========================================================================#

class Transcoder:
    """This class represents any TRANSCODER object transmitted through ComfyUI's node system."""

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str = "",
                        *,# keyword-only arguments #
                        filename  : str = "",
                        device    : torch.device = None,
                        dtype     : torch.dtype  = None,
                        ) -> "Transcoder":
        """
        Creates a new `Transcoder` instance from a state dictionary.
        Args:
            state_dict: A dictionary containing the tensor parameters of a Transcoder model.
            prefix    : The prefix that indicates where the model is located within state_dict.
            filename  : The name of the file from which state_dict was loaded.
            device    : The device where the model will be loaded.
                        If no specified, the default ComfyUI VAE device will be used.
            dtype     : The data type used for the model.
                        If no specified, the default ComfyUI VAE data type will be used.
        """
        filename = os.path.basename(filename)

        # try to create the custom Transcoder model
        model, missing_keys = _create_custom_transcoder_model(state_dict, prefix, filename)
        if not model:
            raise ValueError("TRANSCODER: unsupported model type")
        if missing_keys:
            logger.warning(f"Missing TRANSCODER keys: {missing_keys}")

        # determine device and data type where the model will be loaded
        # (use comfyui default VAE device and data type if not specified)
        working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if device is None:
            device = model_management.vae_device()
        if dtype is None:
            dtype = model_management.vae_dtype(device, working_dtypes)

        model.emulate_std_decoderencoder = True
        model.to(device=device, dtype=dtype)
        model.freeze()
        logger.info(f"TRANSCODER model device: {device}, dtype: {dtype}")
        return cls(model=model, model_device=device, model_dtype=dtype)


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
                 model                      = None,
                 model_device: torch.device = None,
                 model_dtype : torch.dtype  = None,
                 decoder     : VAE          = None,
                 encoder     : VAE          = None,
                 gaussian_blur_sigma: float = 0.0 ,
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
        unrounded_kernel_size = ((sigma - 0.8) / 0.3 + 1) * 2 + 1
        kernel_size           = int(unrounded_kernel_size)
        if kernel_size < unrounded_kernel_size:
            kernel_size += 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        # apply gaussian blur using the calculated kernel_size and sigma
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
        tensor = tensor.permute(0, 2, 3, 1)
        return tensor



