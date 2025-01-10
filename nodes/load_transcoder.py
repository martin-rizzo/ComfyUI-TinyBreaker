"""
File    : load_transcoder.py
Purpose : Node to load transcoder models used to convert images from one latent space to another.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 30, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.utils
import comfy.model_management
from   .utils.directories          import VAE_DIR
from   .xcomfy.transcoder          import Transcoder


class LoadTranscoder:
    TITLE       = "💪TB | Load Transcoder"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load a transcoder model used to convert images from one latent space to another."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcoder_name": (VAE_DIR.get_filename_list(), {"tooltip": "The transcoder model to load."}),
                }
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_transcoder"
    RETURN_TYPES    = ("TRANSCODER",)
    OUTPUT_TOOLTIPS = ("The loaded transcoder model.")

    @classmethod
    def load_transcoder(cls, transcoder_name):
        safetensors_path = VAE_DIR.get_full_path(transcoder_name)
        state_dict       = comfy.utils.load_torch_file(safetensors_path, safe_load=True)
        transcoder       = Transcoder.from_state_dict(state_dict)
        return (transcoder,)
