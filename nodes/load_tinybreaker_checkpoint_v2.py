"""
File    : load_tinybreaker_checkpoint_v2.py
Purpose : Node to load TinyBreaker checkpoints.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 1, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .xcomfy.model      import Model
from .xcomfy.vae        import VAE
from .xcomfy.clip       import CLIP
from .xcomfy.transcoder import Transcoder
from .utils.directories import TINYBREAKER_CHECKPOINTS_DIR
from .core.genparams    import GenParams

_AUTOMATIC        = "auto"
_VAE_TYPE_FAST    = "fast"
_VAE_TYPE_QUALITY = "high quality"
_VAE_TYPES = [ _AUTOMATIC, _VAE_TYPE_QUALITY ]


class LoadTinyBreakerCheckpointV2:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "ckpt_name": (cls.ckpt_names(), {"tooltip": "The TinyBreaker checkpoint to load."}),
            "vae_type" : (_VAE_TYPES      , {"tooltip": "The VAE quality to use.",
                                             "default": _AUTOMATIC
                                            }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "TRANSCODER", "MODEL"        , "CLIP"        , "VAE", "GENPARAMS")
    RETURN_NAMES = ("MODEL", "CLIP", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "VAE", "GENPARAMS")
    OUTPUT_TOOLTIPS = ("The model used for denoising latent images.",
                       "The CLIP model used for embedding text prompts."
                       "The transcoder model used for converting latent images from base to refiner.",
                       "The model used for refining latent images.",
                       "The CLIP model used for embedding text prompts during refining.",
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "Generation parameters extracted from the metadata of the loaded checkpoint.",
                       )

    def load_checkpoint(self, ckpt_name, vae_type):

        # resolve the automatic settings
        if  vae_type == _AUTOMATIC:
            vae_type =  _VAE_TYPE_FAST

        # determine the VAE model prefix (used in the safetensors file)
        if   vae_type  == _VAE_TYPE_FAST:
             vae_prefix = "first_stage_model"
        elif vae_type  == _VAE_TYPE_QUALITY:
             vae_prefix = "first_stage_hqmodel"
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")

        ckpt_full_path = TINYBREAKER_CHECKPOINTS_DIR.get_full_path_or_raise(ckpt_name)
        state_dict     = TINYBREAKER_CHECKPOINTS_DIR.load_state_dict_or_raise(ckpt_name)

        genparams      =  GenParams.from_safetensors_metadata(ckpt_full_path)
        model          =      Model.from_state_dict( state_dict, prefix="base.diffusion_model"   , resolution=1024   )
        transcoder     = Transcoder.from_state_dict( state_dict, prefix="transcoder"             , filename=ckpt_name)
        refiner_model  =      Model.from_state_dict( state_dict, prefix="refiner.diffusion_model")
        refiner_clip   =       CLIP.from_state_dict( state_dict, prefix="refiner.conditioner"    , clip_type="stable_diffusion")
        vae            =        VAE.from_state_dict( state_dict, prefix=vae_prefix               , filename=ckpt_name)

        return (model, None, transcoder, refiner_model, refiner_clip, vae, genparams)


    #__ internal functions ________________________________

    @staticmethod
    def ckpt_names():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()


