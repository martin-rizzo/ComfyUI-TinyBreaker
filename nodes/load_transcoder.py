"""
File    : load_transcoder.py
Purpose : Node to load a transcoder model to convert images from one latent space to another.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 30, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.utils
import comfy.model_management
from   .utils.directories          import VAE_DIR
from   .xcomfy.transcoder          import Transcoder
from   .core.tiny_transcoder_model import TinyTranscoderModel


#-- CLASSES ------------------------------------#

class LoadTranscoder:
    TITLE       = "xPixArt | Load Transcoder"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Load a transcoder model to convert images from one latent space to another."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcoder_name": (VAE_DIR.get_filename_list(), {"tooltip": "The transcoder model to load."}),
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION = "load"
    RETURN_TYPES    = ("TRANSCODER",)
    OUTPUT_TOOLTIPS = ("The loaded transcoder model.")

    @classmethod
    def load(cls, transcoder_name):
        device = comfy.model_management.vae_device()

        safetensors_path = VAE_DIR.get_full_path(transcoder_name)
        state_dict       = comfy.utils.load_torch_file(safetensors_path, safe_load=True)
        transcoder_model = TinyTranscoderModel.from_state_dict(state_dict, prefix="")
        transcoder_model = transcoder_model.to(device)
        transcoder_model.freeze()
        transcoder = Transcoder.from_model(transcoder_model)
        return (transcoder,)
