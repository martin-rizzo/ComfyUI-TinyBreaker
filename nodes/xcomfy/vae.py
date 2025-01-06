"""
File    : xconfy/vae.py
Purpose : The standard VAE object transmitted through ComfyUI's node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 10, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.sd
from comfy                       import model_management
from ..utils.safetensors         import normalize_safetensors_prefix
from ..utils.system              import logger
from ..core.autoencoder_model_ex import AutoencoderModelEx




def _custom_vae_model_from_state_dict(state_dict, prefix, config, vae_wrapper):
    """
    Creates a custom VAE model from the given state_dict and config.
    """
    if config is not None:
        return None

    if AutoencoderModelEx.detect_prefix(state_dict, prefix) is not None:
        logger.info("Detected custom AutoencoderModelEx model")
        vae_model = AutoencoderModelEx.from_state_dict(state_dict)
        vae_wrapper.latent_channels = state_dict["decoder.conv_in.weight"].shape[1]
        return vae_model

    return None


def _is_likely_custom_vae_model(state_dict, config):
    """
    Returns `True` if the state_dict likely represents a custom VAE model handled by this project.

    The primary purpose of this check is to quickly filter out models that
    are known to be incompatible, avoiding more costly processing and possible
    mismatches.
    """
    CONFYUI_STANDARD_TENSOR_NAMES = [
        "decoder.mid.block_1.mix_factor",                     # <- VIDEO (?)
        "taesd_decoder.1.weight",                             # <- Tiny Autoencoder
        "vquantizer.codebook.weight",                         # <- VQGan (Stage A of stable cascade)
        "backbone.1.0.block.0.1.num_batches_tracked",         # <- effnet (encoder for Stage C of stable cascade)
        "blocks.11.num_batches_tracked",                      # <- previewer (decoder for Stage C of stable cascade)
        "encoder.backbone.1.0.block.0.1.num_batches_tracked", # <- combined effnet and previewer for stable cascade
        ]
    if config is not None:
        return False
    elif any( tensor_name in state_dict for tensor_name in CONFYUI_STANDARD_TENSOR_NAMES ):
        return False
    elif "decoder.conv_in.weight"                    in state_dict and \
         "encoder.down.2.downsample.conv.weight" not in state_dict and \
         "decoder.up.3.upsample.conv.weight"     not in state_dict:
        # stable diffusion x4 upscaler VAE
        return False
    else:
        return True


class VAE(comfy.sd.VAE):
    """
    A class representing a Variational Autoencoder (VAE).

    This class provides a bridge to the `VAE` class definided in comfy.sd module.
    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    """

    def __init__(self, sd=None, device=None, config=None, dtype=None, offload_device = None):
        """
        Custom initialization method for the VAE class.

        This initializator overrides the default initialization of comfy.sd.VAE,
        adding support for custom autoencoder models.

        Args:
            sd                     (dict): A dictionary containing the state_dict of the VAE.
            device         (torch.device): The device where the model will be loaded when it's active and being used.
            config                 (dict): A dictionary containing the configuration of the VAE.
            dtype           (torch.dtype): The data type to which the tensors of the model will be converted.
            offload_device (torch.device): The device where the model will be offloaded when it's not active.
        """

        #if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys(): #diffusers format
        #    sd = diffusers_convert.convert_vae_state_dict(sd)

        self.memory_used_encode      = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) #These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode      = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
        self.downscale_ratio         = 8
        self.upscale_ratio           = 8
        self.latent_channels         = 4
        self.latent_dim              = 2
        self.output_channels         = 3
        self.process_input           = lambda image: image * 2.0 - 1.0
        self.process_output          = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        self.working_dtypes          = [torch.bfloat16, torch.float32]
        self.downscale_index_formula = None
        self.upscale_index_formula   = None
        self.first_stage_model       = None

        if _is_likely_custom_vae_model(sd, config):
            self.first_stage_model = _custom_vae_model_from_state_dict(sd, "", config, self)

        # if the autoencoder model could be created (first_stage_model)
        # then a custom initialization based on ComfyUI is used
        if self.first_stage_model:

            self.first_stage_model.eval()
            missing_keys, unexpected_keys = self.first_stage_model.load_state_dict(sd, strict=False)
            if missing_keys:
                logger.warning(f"Missing VAE keys {missing_keys}")
            if unexpected_keys:
                logger.debug(f"Leftover VAE keys {unexpected_keys}")

            if device is None:
                device = model_management.vae_device()
            if dtype is None:
                dtype  = model_management.vae_dtype(device, self.working_dtypes)
            if offload_device is None:
                offload_device = model_management.vae_offload_device()

            self.device        = device
            self.vae_dtype     = dtype
            self.output_device = model_management.intermediate_device()

            self.first_stage_model.to(self.vae_dtype)
            self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
            logger.info(f"VAE load device: {self.device}, offload device: {offload_device}, dtype: {self.vae_dtype}")

        # if the autoencoder model could NOT be created
        # then use the ComfyUI's default initialization
        else:
            super().__init__(sd, device, config, dtype)



    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str  = "",
                        device           = None,
                        dtype            = None
                        ) -> "VAE":
        """
        Creates an instance of VAE from a given state dictionary.
        Args:
            state_dict   (dict): A dictionary containing the state of the VAE model.
            prefix       (str) : A string indicating a prefix to filter the keys in the state_dict. Defaults to "".
            device       (str) : The device on which the VAE should be loaded. Defaults to None.
            config       (dict): A dictionary containing configuration parameters for the VAE. Defaults to None.
            dtype (torch.dtype): The data type of the tensors in the VAE model. Defaults to None.
        """

        # always ensure that `prefix` is normalized
        prefix = normalize_safetensors_prefix(prefix)

        # if a prefix is provided, then only the corresponding part needs to be loaded
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}

        # load the VAE model using the custom initialization
        vae = cls(state_dict, device=device, config=None, dtype=dtype)
        return vae

