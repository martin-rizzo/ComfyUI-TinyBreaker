"""
File    : load_tinybreaker_checkpoint.py
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
from .xcomfy.model      import Model
from .xcomfy.vae        import VAE
from .xcomfy.clip       import CLIP
from .xcomfy.transcoder import Transcoder
from .utils.directories import TINYBREAKER_CHECKPOINTS_DIR
from .core.gen_params   import GenParams

_VAE_OPTIONS        = ["fast", "high quality"]
_DEFAULT_VAE_OPTION = "fast"


class LoadCheckpoint:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (cls.checkpoint_list(), {"tooltip": "The TinyBreaker checkpoint to load."}),
                "vae"      : (_VAE_OPTIONS         , {"tooltip": "The VAE quality to use.",
                                                      "default": _DEFAULT_VAE_OPTION}),
                }
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "TRANSCODER", "MODEL"        , "CLIP"        , "GENPARAMS")
    RETURN_NAMES = ("MODEL", "VAE", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "GENPARAMS")
    OUTPUT_TOOLTIPS = ("The model used for denoising latent images.",
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "The transcoder model used for converting latent images from base to refiner.",
                       "The model used for refining latent images.",
                       "The CLIP model used for embedding text prompts during refining.",
                       "Generation parameters extracted from the metadata of the loaded checkpoint.",
                       )

    def load_checkpoint(self, ckpt_name, vae):
        ckpt_path  = TINYBREAKER_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        vae_prefix = "first_stage_hdmodel" if vae == "high quality" else "first_stage_model"
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        genparams      = GenParams()  # GenParams.from_metadata("sigma", 2048)
        model_obj      = Model.from_state_dict(state_dict, prefix="base.diffusion_model", resolution=1024)
        vae_obj        = VAE.from_state_dict(state_dict, prefix=vae_prefix, filename=ckpt_name)
        transcoder_obj = Transcoder.from_state_dict(state_dict, prefix="transcoder", filename=ckpt_name)
        refiner_model  = Model.from_state_dict(state_dict, prefix="refiner.diffusion_model")
        refiner_clip   = CLIP.from_state_dict(state_dict, prefix="refiner.conditioner", type="stable_diffusion")
        return (model_obj, vae_obj, transcoder_obj, refiner_model, refiner_clip, genparams)


    #__ internal functions ________________________________

    @staticmethod
    def checkpoint_list():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()


