"""
File    : xconfy/vae.py
Purpose : The standard VAE object transmitted through ComfyUI's node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 10, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.sd
from comfy                            import model_management
from ..utils.safetensors              import normalize_safetensors_prefix
from ..utils.system                   import logger
from ..core.autoencoder_model_ex      import AutoencoderModelEx
from ..core.tiny_autoencoder_model_ex import TinyAutoencoderModelEx



def _create_custom_vae_model(state_dict : dict,
                             prefix     : str,
                             filename   : str,
                             vae_wrapper: "VAE"
                             ) -> tuple[object, list]:
    """
    Main function to create a custom VAE model from the given state_dict.
    Here must be added the code to detect and instantiate any custom VAE model.

    Args:
        state_dict : The dictionary containing the model's parameters.
        prefix     : A prefix indicating which of the tensors in state_dict belong to the model.
        filename   : The name of the file from which state_dict was loaded.
        vae_wrapper: The ComfyUI wrapper object. This is used by ComfyUI to store
                     the model's memory usage and other properties.
    Returns:
        A tuple containing the VAE model and a list of missing keys (if any)
    """

    # state_dict == None is a small hack to indicate that
    # the wrapper should be initialized with default values,
    # (these values can be overridden later by any model)
    if not state_dict:
        vae_wrapper.memory_used_encode      = lambda shape, dtype: (1767 * shape[2] * shape[3]     ) * model_management.dtype_size(dtype) # <- for AutoencoderKL and need tweaking (should be lower)
        vae_wrapper.memory_used_decode      = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
        vae_wrapper.downscale_ratio         = 8
        vae_wrapper.upscale_ratio           = 8
        vae_wrapper.latent_channels         = 4
        vae_wrapper.latent_dim              = 2
        vae_wrapper.output_channels         = 3
        vae_wrapper.process_input           = lambda image: image * 2.0 - 1.0
        vae_wrapper.process_output          = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        vae_wrapper.working_dtypes          = [torch.float16, torch.bfloat16, torch.float32]
        vae_wrapper.downscale_index_formula = None
        vae_wrapper.upscale_index_formula   = None
        return

    # detect the classic AutoencoderKL model used by Stable Diffusion
    if AutoencoderModelEx.detect_prefix(state_dict, prefix) is not None:
        logger.info(f"Loading AutoencoderModelEx from '{filename}'")
        vae_model, config, missing_keys, _ = \
            AutoencoderModelEx.from_state_dict(state_dict, prefix)
        vae_wrapper.latent_channels = config["latent_channels"] # <- overrides latent channels
        vae_model.freeze()
        return vae_model, missing_keys

    # detect the Tiny Autoencoder model by @madebyollin (https://github.com/madebyollin/taesd)
    elif TinyAutoencoderModelEx.detect_prefix(state_dict, prefix) is not None:
        logger.info(f"Loading TinyAutoencoderModelEx from '{filename}'")
        vae_model, config, missing_keys, _ = \
            TinyAutoencoderModelEx.from_state_dict(state_dict, prefix, filename=filename)
        vae_model.emulate_std_autoencoder = True
        vae_wrapper.latent_channels = config["latent_channels"] # <- overrides latent channels
        vae_model.freeze()
        return vae_model, missing_keys

    return None, []


def _should_use_custom_code(state_dict, config):
    """
    Returns `True` if the state_dict likely represents a custom model handled by this project.
    The primary purpose of this check is to quickly filter out models that are
    known to be incompatible with this project, avoiding possible mismatches.
    """
    CONFYUI_STANDARD_TENSOR_NAMES = [
        "decoder.mid.block_1.mix_factor",                     # <- VIDEO (?)
        "vquantizer.codebook.weight",                         # <- VQGan (Stage A of stable cascade)
        "backbone.1.0.block.0.1.num_batches_tracked",         # <- effnet (encoder for Stage C of stable cascade)
        "blocks.11.num_batches_tracked",                      # <- previewer (decoder for Stage C of stable cascade)
        "encoder.backbone.1.0.block.0.1.num_batches_tracked", # <- combined effnet and previewer for stable cascade
        ]

    if config is not None:
        # custom code doesn´t handle config files
        return False

    if any( tensor_name in state_dict for tensor_name in CONFYUI_STANDARD_TENSOR_NAMES ):
        # these models are evidently not custom models
        return False

    if "decoder.conv_in.weight"                      in state_dict and \
         "encoder.down.2.downsample.conv.weight" not in state_dict and \
         "decoder.up.3.upsample.conv.weight"     not in state_dict:
        # stable diffusion x4 upscaler VAE is not handled by this project yet
        return False

    # at this point we assume that
    # the model can be handled by this project
    return True


class VAE(comfy.sd.VAE):
    """
    A class representing a Variational Autoencoder (VAE).

    This class provides a bridge to the `VAE` class definided in comfy.sd module.
    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    """

    def __init__(self,
                 state_dict: dict = None,
                 *,
                 config        : dict         = None,
                 filename      : str          = "",
                 dtype         : torch.dtype  = None,
                 device        : torch.device = None,
                 offload_device: torch.device = None):
        """
        Custom initialization method for the VAE class.

        This initializator overrides the default initialization of comfy.sd.VAE,
        adding support for custom autoencoder models.

        Args:
            state_dict             (dict): A dictionary containing the tensors of the VAE model.
            config                 (dict): A dictionary containing the configuration of the VAE.
            filename                (str): The name of the file where the VAE is stored (some models use this for autodetection)
            device         (torch.device): The device where the model will be loaded when it's active and being used.
            dtype           (torch.dtype): The data type to which the tensors of the model will be converted.
            offload_device (torch.device): The device where the model will be offloaded when it's not active.
        """
        # ATTENTION!:
        # this arguments must be in the same order as in `comfy.sd.VAE.__init__(...)`
        super_args = [state_dict, device, config, dtype]

        # set default values. e.g. self.memory_used_encode, self.downscale_ratio, self.latent_channels, etc.
        _create_custom_vae_model(None, None, None, vae_wrapper = self)

        # try to create a custom VAE model from the state_dict
        if _should_use_custom_code(state_dict, config):
            model, missing_keys = _create_custom_vae_model(state_dict, "", filename, vae_wrapper = self)

        # if a custom VAE model was created,
        # then use this custom initialization code based on ComfyUI
        if model:

            self.first_stage_model = model
            if missing_keys:
                logger.warning(f"Missing TRANSCODER keys: {missing_keys}")

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

        # if a custom VAE model could NOT be created,
        # then use the actual ComfyUI initialization
        else:
            super().__init__(*super_args)



    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str  = "",
                        *,# keyword-only arguments #
                        filename  : str  = "",
                        device           = None,
                        dtype            = None
                        ) -> "VAE":
        """
        Creates an instance of VAE from a given state dictionary.
        Args:
            state_dict: A dictionary containing the tensor parameters of the model.
            prefix    : A prefix indicating which of the tensors in state_dict belong to the model.
            filename  : The name of the file from which state_dict was loaded.
            device    : The device on which the VAE should be loaded. Defaults to None.
            dtype     : The data type of the tensors in the VAE model. Defaults to None.
        """

        # always ensure that `prefix` is normalized
        prefix = normalize_safetensors_prefix(prefix)

        # if a prefix is provided, then only the corresponding part needs to be loaded
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}

        # load the VAE model using the custom initialization
        vae = cls(state_dict, config=None, filename=filename, device=device, dtype=dtype)
        return vae

