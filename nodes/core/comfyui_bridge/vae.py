"""
File    : xcomfy/vae.py

Purpose : The standard VAE object transmitted through ComfyUI's node system.
          This VAE object is directly derived from `comfy.sd.VAE`, extending
          it to support my custom Autoencoder and TinyAutoencoder code.

Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 10, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.sd
from comfy                               import model_management
from ..safetensors                       import normalize_prefix
from ..system                            import logger
from ..models.autoencoder_model_ex       import AutoencoderModelEx
from ..models.tiny_autoencoder_model_ex  import TinyAutoencoderModelEx
from ..models.combined_autoencoder_model import CombinedAutoencoderModel
from .helpers.ops                        import comfy_ops_disable_weight_init



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
    # ATTENTION:
    #   using `state_dict == None` is a small hack to indicate that the wrapper should be initialized with default values!
    #   default values can be found at: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.27/comfy/sd.py#L258
    #  (these values can be overridden later during model initialization when state_dict is not None)
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
        vae_wrapper.working_dtypes          = [torch.bfloat16, torch.float32] # not torch.float16 (?)
        vae_wrapper.disable_offload         = False
        vae_wrapper.downscale_index_formula = None
        vae_wrapper.upscale_index_formula   = None
        return

    # vae type auto-detection
    standard_vae_detected = AutoencoderModelEx.detect_prefix(state_dict, prefix)     is not None
    tiny_vae_detected     = TinyAutoencoderModelEx.detect_prefix(state_dict, prefix) is not None


    # Combined Autoencoder Model
    # - This is an experimental hybrid model that combines both Standard and Tiny Autoencoder models.
    # - Example: the encoder can be `Tiny` and the decoder can be `Standard`, or vice versa.
    if standard_vae_detected and tiny_vae_detected:
        logger.info(f"Loading a Combined Autoencoder Model from '{filename}'")

        standard_model, config, standard_missing_keys, _ = \
            AutoencoderModelEx.from_state_dict(state_dict, prefix,
                                               nn = comfy_ops_disable_weight_init) # <- replaces `torch.nn` with ComfyUI's version
        tiny_model, config, tiny_missing_keys, _ = \
            TinyAutoencoderModelEx.from_state_dict(state_dict, prefix,
                                                   filename = filename,
                                                   nn = comfy_ops_disable_weight_init) # <- replaces `torch.nn` with ComfyUI's version
        vae_wrapper.latent_channels = config["latent_channels"] # <- overrides latent channels
        tiny_model.emulate_std_autoencoder = True               # <- force tiny model to behave like a standard autoencoder
        vae_model = CombinedAutoencoderModel(standard_model, tiny_model)
        vae_model.freeze()
        return vae_model, [*standard_missing_keys, *tiny_missing_keys]


    # Standard Autoencoder Model
    # - This is the standard Variational Autoencoder (VAE)
    #   commonly utilized in Stable Diffusion (SD) models.
    if standard_vae_detected:
        logger.info(f"Loading a Standard Autoencoder Model from '{filename}'")

        vae_model, config, missing_keys, _ = \
            AutoencoderModelEx.from_state_dict(state_dict, prefix,
                                               nn = comfy_ops_disable_weight_init) # <- replaces `torch.nn` with ComfyUI's version
        vae_wrapper.latent_channels = config["latent_channels"] # <- overrides latent channels
        vae_model.freeze()
        return vae_model, missing_keys


    # Tiny Autoencoder Model
    # - This is a distilled model using a specialized architecture developed by @madebyollin.
    # - It's designed for exceptionally fast encoding and decoding operations,
    #   while also maintaining a significantly reduced model size.
    # - Refer to the original repository for more details: https://github.com/madebyollin/taesd
    if tiny_vae_detected:
        logger.info(f"Loading Tiny Autoencoder Model from '{filename}'")

        vae_model, config, missing_keys, _ = \
            TinyAutoencoderModelEx.from_state_dict(state_dict, prefix,
                                                   filename = filename,
                                                   nn = comfy_ops_disable_weight_init) # <- replaces `torch.nn` with ComfyUI's version
        vae_wrapper.latent_channels = config["latent_channels"] # <- overrides latent channels
        vae_wrapper.disable_offload = True                      # <- try to keep the model in GPU memory
        vae_model.emulate_std_autoencoder = True                # <- force tiny model to behave like a standard autoencoder
        vae_model.freeze()
        return vae_model, missing_keys

    # Unknown model
    # return `None` to force ComfyUI to load the model using its default mechanism
    return None, []



def _should_use_custom_code(state_dict, config):
    """
    Returns `True` if the state_dict likely represents a custom model handled by this project.
    The primary purpose of this check is to quickly filter out models that are
    known to be incompatible with this project, avoiding possible mismatches.
    """
    COMFYUI_STANDARD_TENSOR_NAMES = [
        "decoder.mid.block_1.mix_factor",                     # <- VIDEO (?)
        "vquantizer.codebook.weight",                         # <- VQGan (Stage A of stable cascade)
        "backbone.1.0.block.0.1.num_batches_tracked",         # <- effnet (encoder for Stage C of stable cascade)
        "blocks.11.num_batches_tracked",                      # <- previewer (decoder for Stage C of stable cascade)
        "encoder.backbone.1.0.block.0.1.num_batches_tracked", # <- combined effnet and previewer for stable cascade
        ]

    if config is not None:
        # custom code doesn´t handle config files
        return False

    if any( tensor_name in state_dict for tensor_name in COMFYUI_STANDARD_TENSOR_NAMES ):
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


#===========================================================================#
#/////////////////////////////////// VAE ///////////////////////////////////#
#===========================================================================#

class VAE(comfy.sd.VAE):
    """
    A class representing a Variational Autoencoder (VAE).

    This class provides a bridge to the `VAE` class definided in comfy.sd module.
    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    """

    def __init__(self,
                 state_dict: dict,
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
        assert state_dict, "state_dict is required in the constructor of the VAE class"

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
                logger.warning(f"Missing VAE keys: {missing_keys}")

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
        assert state_dict, "state_dict is required in VAE.from_state_dict(..)"

        # always ensure that `prefix` is normalized
        prefix = normalize_prefix(prefix)

        # if a prefix is provided, then only the corresponding part needs to be loaded
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}

        # load the VAE model using the custom initialization
        vae = cls(state_dict, config=None, filename=filename, device=device, dtype=dtype)
        return vae

