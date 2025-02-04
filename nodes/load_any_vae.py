"""
File    : load_any_vae.py
Purpose : Node to load any VAE model including `Tiny AutoEncoder` (TAESD) variants.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 16, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .utils.directories import VAE_DIR
from .xcomfy.vae        import VAE
from comfy.utils        import load_torch_file


class LoadAnyVAE:
    TITLE       = "💪TB | Load Any VAE"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load VAE models including `Tiny AutoEncoder` (TAESD) variants."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (VAE_DIR.get_filename_list(), ),
                }
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_any_vae"
    RETURN_TYPES = ("VAE",)

    def load_any_vae(self, vae_name):
        # load the model's state dictionary
        vae_path   = VAE_DIR.get_full_path_or_raise(vae_name)
        state_dict = load_torch_file(vae_path)
        # return a new instance of the VAE class from the loaded state dictionary
        vae = VAE.from_state_dict(state_dict, filename=vae_name)
        return (vae,)

