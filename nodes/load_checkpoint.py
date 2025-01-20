"""
File    : load_checkpoint.py
Purpose : Node to load TinyBreaker checkpoints.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 1, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.utils
from .xcomfy.model               import Model
from .xcomfy.clip                import CLIP
from .xcomfy.transcoder          import Transcoder
from .utils.directories          import TINYBREAKER_CHECKPOINTS_DIR


class LoadCheckpoint:
    TITLE       = "💪TB | Load Checkpoint"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (TINYBREAKER_CHECKPOINTS_DIR.get_filename_list(), {"tooltip": "The TinyBreaker checkpoint to load. (PixArt sigma checkpoints are also supported)"}),
                }
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "CLIP", "TRANSCODER", "MODEL"        , "CLIP"        , "STRING"  )
    RETURN_NAMES = ("MODEL", "VAE", "CLIP", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "METADATA")

    def load_checkpoint(self, ckpt_name):

        ckpt_path  = TINYBREAKER_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        metadata      = None  # Metadata.from_predefined("sigma", 2048)
        model         = Model.from_state_dict(state_dict, prefix="base.diffusion_model", resolution=1024)
        vae           = None  # VAE.from_state_dict(state_dict, prefix="")
        clip          = None
        transcoder    = Transcoder.from_state_dict(state_dict, prefix="transcoder", filename=ckpt_name)
        refiner_model = Model.from_state_dict(state_dict, prefix="refiner.diffusion_model")
        refiner_clip  = CLIP.from_state_dict(state_dict, prefix="refiner.conditioner", type="stable_diffusion")
        return (model, vae, clip, transcoder, refiner_model, refiner_clip, metadata)
