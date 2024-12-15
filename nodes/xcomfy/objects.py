"""
File     : xconfy/objects.py
Purpose  : The ComfyUI objects transmitted across connected nodes.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 10, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 File Summary
 ============
  - Model_pack: The object transmitted through `MODEL -> model` threads
  - VAE       : The object transmitted through `VAE -> vae` threads
  - Meta_pack : The object transmitted through `META -> meta` threads

"""
import torch
import comfy.sd
import comfy.utils
import comfy.model_patcher
from   typing         import Union
from   comfy          import model_management
from   .bridge        import create_model_from_safetensors
from   ..utils.system import logger


#===========================================================================#
class Model_pack(comfy.model_patcher.ModelPatcher):
    # el objeto que es transmitido por los hilos "MODEL -> model"
    # debe ser compatible con la siguiente estructura:
    #  - class ModelPatcher
    #      - class BaseModel
    #                .diffusion_model = PixArtMS(..)
    #
    #  ModelPatcher: [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py]

    def __init__(self,
                 model,
                 load_device,
                 offload_device,
                 size=0,
                 weight_inplace_update=False
                 ):
        super().__init__(model,
                         load_device=load_device,
                         offload_device=offload_device,
                         size=size,
                         weight_inplace_update=weight_inplace_update
                         )

    # pequeña emulacion al comportamiento de "load_unet_state_dict(sd)"
    # - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py
    @classmethod
    def from_safetensors(cls,
                         safetensors_path,
                         prefix="",
                         weight_inplace_update=False,
                         dtype=None
                         ):


        load_device    = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        model = create_model_from_safetensors(safetensors_path,
                                              prefix         = prefix,
                                              load_device    = load_device,
                                              offload_device = offload_device)
        model.diffusion_model.freeze()

        # envolver al modelo en el nuevo objeto (derivado de ModelPatcher)
        return Model_pack(
            model,
            size = 0,
            load_device           = load_device,
            offload_device        = offload_device,
            weight_inplace_update = weight_inplace_update
            )


#===========================================================================#
class VAE(comfy.sd.VAE):
    """
    A class representing a Variational Autoencoder (VAE).

    This class provides a bridge to the `VAE` class definided in comfy.sd module.
    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    """

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str         = "",
                        device    : str         = None,
                        config    : dict        = None,
                        dtype     : torch.dtype = None
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
        if not prefix:
            return cls(sd=state_dict, device=device, config=config, dtype=dtype)

        # ensure that prefix always ends with a dot '.'
        if not prefix.endswith('.'):
            prefix += '.'

        # if a prefix is provided, then only the corresponding part needs to be loaded
        sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return cls(sd=sd, device=device, config=config, dtype=dtype)


#===========================================================================#
class Meta_pack(dict):

    def from_predefined(mode, size):
        return None


