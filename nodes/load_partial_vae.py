"""
File    : load_partial_vae.py
Purpose : Node to load a partial AutoEncoder (VAE) for testing purposes.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 4, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .utils.directories import VAE_DIR
from .xcomfy.vae        import VAE
from comfy.utils        import load_torch_file as comfy_load_torch_file

_COMPONENT_TO_REMOVE_BY_MODE  = {
    "encoder only": "decoder", # remove 'decoder'
    "decoder only": "encoder", # remove 'encoder'
    "both"        : ""         # don't remove anything
}
_DEFAULT_MODE = "both"


class LoadPartialVAE:
    TITLE       = "💪TB | Load Partial VAE"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Partially load a Variational AutoEncoder (VAE), specifically either the encoder or the decoder. This node serves primarily for testing purposes."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (cls.vae_list() , {"tooltip": "The name of the VAE to load."}),
                "mode"    : (cls.mode_list(), {"tooltip": "The mode to load the VAE.",
                                               "default": _DEFAULT_MODE}),
                },
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_partial_vae"
    RETURN_TYPES = ("VAE",)

    def load_partial_vae(self, vae_name, mode):
        vae_path   = VAE_DIR.get_full_path(vae_name)
        state_dict = comfy_load_torch_file(vae_path)

        # remove the component that the user doesn't want to load
        component_to_remove = _COMPONENT_TO_REMOVE_BY_MODE[mode]
        state_dict = self.remove_component(state_dict, component_to_remove)

        # create the new VAE object from the modified state dict
        vae = VAE.from_state_dict(state_dict)
        return (vae,)



    #__ internal functions ________________________________

    @staticmethod
    def vae_list():
        return VAE_DIR.get_filename_list()


    @staticmethod
    def mode_list():
        return list(_COMPONENT_TO_REMOVE_BY_MODE.keys())


    @staticmethod
    def remove_component(state_dict, component_to_remove):
        if not component_to_remove:
            return state_dict
        # filter out the component to remove from the state dict
        return {key: tensor for key, tensor in state_dict.items() if component_to_remove not in key}

