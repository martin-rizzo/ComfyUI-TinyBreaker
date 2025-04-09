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
from .xcomfy.model        import Model
from .xcomfy.vae          import VAE
from .xcomfy.clip         import CLIP
from .xcomfy.transcoder   import Transcoder
from .utils.directories   import TINYBREAKER_CHECKPOINTS_DIR
from .functions.genparams import GenParams

_AUTOMATIC        = "auto"
_VAE_TYPE_FAST    = "fast"
_VAE_TYPE_QUALITY = "high_quality"
_VAE_TYPES = [ _AUTOMATIC, _VAE_TYPE_FAST, _VAE_TYPE_QUALITY ]


class LoadTinyBreakerCheckpointV2:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "ckpt_name"        : (cls.ckpt_names(), {"tooltip": "The TinyBreaker checkpoint to load."}),
            "vae_type"         : (_VAE_TYPES      , {"tooltip": "The VAE quality to use.",
                                                     "default": _AUTOMATIC
                                                    }),
            "upscaler_vae_type": (_VAE_TYPES      , {"tooltip": "The VAE quality to use.",
                                                     "default": _AUTOMATIC
                                                    }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "TRANSCODER", "MODEL"        , "CLIP"        , "VAE"         , "GENPARAMS")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "UPSCALER_VAE", "METADATA" )
    OUTPUT_TOOLTIPS = ("The model used for denoising latent images.",
                       "The CLIP model used for embedding text prompts."
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "The transcoder model used for converting latent images from base to refiner.",
                       "The model used for refining latent images.",
                       "The CLIP model used for embedding text prompts during refining.",
                       "The VAE model used during the upscaling process.",
                       "Generation parameters extracted from the metadata of the loaded checkpoint.",
                       )

    def load_checkpoint(self, ckpt_name, vae_type, upscaler_vae_type):

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
        model_type     = self.get_model_type(state_dict)

        if model_type == "prototype0":
            metadata       =  GenParams.from_safetensors_metadata(ckpt_full_path)
            model          =      Model.from_state_dict( state_dict, prefix="base.diffusion_model"   , resolution=1024   )
            transcoder     = Transcoder.from_state_dict( state_dict, prefix="transcoder"             , filename=ckpt_name)
            refiner_model  =      Model.from_state_dict( state_dict, prefix="refiner.diffusion_model")
            refiner_clip   =       CLIP.from_state_dict( state_dict, prefix="refiner.conditioner"    , clip_type="stable_diffusion")
            vae            =        VAE.from_state_dict( state_dict, prefix=vae_prefix               , filename=ckpt_name)
            return (model, None, vae, transcoder, refiner_model, refiner_clip, None, metadata)

        if model_type == "prototype1":
            metadata      =  GenParams.from_safetensors_metadata(ckpt_full_path)
            vae            =        VAE.from_state_dict( state_dict, prefix="first_stage_model"        , filename=ckpt_name)
            model          =      Model.from_state_dict( state_dict, prefix="base.diffusion_model"     , resolution=1024   )
            transcoder     = Transcoder.from_state_dict( state_dict, prefix="transcoder"               , filename=ckpt_name)
            refiner_vae    =        VAE.from_state_dict( state_dict, prefix="refiner.first_stage_model", filename=ckpt_name)
            refiner_clip   =       CLIP.from_state_dict( state_dict, prefix="refiner.conditioner"      , clip_type="stable_diffusion")
            refiner_model  =      Model.from_state_dict( state_dict, prefix="refiner.diffusion_model")
            return (model, None, vae, transcoder, refiner_model, refiner_clip, refiner_vae, metadata)


    #__ internal functions ________________________________

    @staticmethod
    def ckpt_names():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()


    @staticmethod
    def get_model_type(state_dict):

        # if the model contains a key that starts with `base.diffusion_model`
        # then it is a model of type "prototype1"
        for key in state_dict.keys():
            if key.startswith("refiner.diffusion_model."):
                return "prototype1"

        # otherwise, it is a model of type "prototype0" (old model)
        return "prototype0"

