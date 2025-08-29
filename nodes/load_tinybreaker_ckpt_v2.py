"""
File    : load_tinybreaker_ckpt_v2.py
Purpose : Node to load TinyBreaker checkpoints.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Aug 29, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams                    import GenParams
from .core.comfyui_bridge.model         import Model
from .core.comfyui_bridge.vae           import VAE
from .core.comfyui_bridge.clip          import CLIP
from .core.comfyui_bridge.transcoder    import Transcoder
from .core.comfyui_bridge.upscale_model import UpscaleModel
from .core.directories import TINYBREAKER_CHECKPOINTS_DIR, UPSCALE_MODELS_DIR
from .core.safetensors import filter_state_dict, update_state_dict, prune_state_dict
from .core.system      import logger
_AUTOMATIC        = "auto"
_VAE_TYPE_FAST    = "fast"
_VAE_TYPE_QUALITY = "high_quality"
_VAE_TYPES = [ _AUTOMATIC, _VAE_TYPE_FAST, _VAE_TYPE_QUALITY ]


class LoadTinyBreakerCkptV2:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load a TinyBreaker checkpoint."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "ckpt_name"        : (cls.ckpt_names(), {"tooltip": "The TinyBreaker checkpoint to load."}),
            "upscaler_name"    : (cls.upscalers() , {"tooltip": "The upscale model to use if none is embedded in the checkpoint."}),
            "vae_type"         : (_VAE_TYPES      , {"tooltip": "The VAE type used during generation. The `high_quality` VAE produces better results but takes longer and uses more VRAM.",
                                                     "default": _AUTOMATIC
                                                    }),
            "upscaler_vae_type": (_VAE_TYPES      , {"tooltip": "The VAE type used during upscaling. A `high_quality` VAE is available but due to its high VRAM consumption, `auto` and `fast` are recommended.",
                                                     "default": _AUTOMATIC
                                                    }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "TRANSCODER", "MODEL"        , "CLIP"        , "UPSCALE_MODEL", "VAE"         , "GENPARAMS")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "UPSCALE_MODEL", "UPSCALER_VAE", "METADATA" )
    OUTPUT_TOOLTIPS = ("The model used for denoising latent images.",
                       "The CLIP model used for embedding text prompts."
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "The transcoder model used for converting latent images from base to refiner.",
                       "The model used for refining latent images.",
                       "The CLIP model used for embedding text prompts during refining.",
                       "The VAE model used during the upscaling process.",
                       "Generation parameters extracted from the metadata of the loaded checkpoint.",
                       )

    def load_checkpoint(self, ckpt_name, upscaler_name, vae_type, upscaler_vae_type):

        # load checkpoint and metadata
        ckpt_full_path = TINYBREAKER_CHECKPOINTS_DIR.get_full_path_or_raise(ckpt_name)
        state_dict     = TINYBREAKER_CHECKPOINTS_DIR.load_state_dict_or_raise(ckpt_name)
        metadata       = GenParams.from_safetensors_metadata(ckpt_full_path)
        model_type     = self.get_model_type(state_dict)
        logger.info(f"Loading '{ckpt_name}' ('{model_type}' checkpoint type).")

        # resolve vae-type automatic settings (quality + upscaler fast)
        vae_type          = _VAE_TYPE_QUALITY if vae_type          == _AUTOMATIC else vae_type
        upscaler_vae_type = _VAE_TYPE_FAST    if upscaler_vae_type == _AUTOMATIC else upscaler_vae_type
        logger.debug(f"Configured VAE type: '{vae_type}'.")
        logger.debug(f"Configured upscaler VAE type: '{upscaler_vae_type}'.")


        if model_type == "TinyBreaker.prototype0":

            # a small hack to be able to configure the quality of the VAE decoder
            hq_vae_state_dict = filter_state_dict( state_dict, "first_stage_hqmodel" )
            if vae_type == _VAE_TYPE_QUALITY:
                prune_state_dict ( state_dict, "first_stage_model", ("decoder", "post_quant_conv") )
                update_state_dict( state_dict, "first_stage_model", hq_vae_state_dict)

            model         =      Model.from_state_dict( state_dict, prefix="base.diffusion_model"   , resolution=1024   )
            transcoder    = Transcoder.from_state_dict( state_dict, prefix="transcoder"             , filename=ckpt_name)
            refiner_model =      Model.from_state_dict( state_dict, prefix="refiner.diffusion_model")
            refiner_clip  =       CLIP.from_state_dict( state_dict, prefix="refiner.conditioner"    , clip_type="stable_diffusion")
            vae           =        VAE.from_state_dict( state_dict, prefix="first_stage_model"      , filename=ckpt_name)
            upscale_model =       None
            upscale_vae   =       None
            return (model, None, vae, transcoder, refiner_model, refiner_clip, upscale_model, upscale_vae, metadata)


        if model_type == "TinyBreaker.prototype1":

            # a small hack to be able to configure the quality of the VAE decoders,
            # assuming that `first_stage_model` is the high-quality VAE while `refiner.first_stage_model` is the fast VAE
            # (prototype1 should always be created in this way)
            hq_vae_state_dict   = filter_state_dict( state_dict, "first_stage_model"        , ("decoder", "post_quant_conv") )
            fast_vae_state_dict = filter_state_dict( state_dict, "refiner.first_stage_model", ("decoder", "post_quant_conv") )
            if vae_type == _VAE_TYPE_FAST:
                prune_state_dict ( state_dict, "first_stage_model", ("decoder", "post_quant_conv") )
                update_state_dict( state_dict, "first_stage_model", fast_vae_state_dict)
            if upscaler_vae_type == _VAE_TYPE_QUALITY:
                prune_state_dict ( state_dict, "refiner.first_stage_model", ("decoder", "post_quant_conv") )
                update_state_dict( state_dict, "refiner.first_stage_model", hq_vae_state_dict)

            model          =        Model.from_state_dict( state_dict, prefix="base.diffusion_model"     , resolution=1024   )
            transcoder     =   Transcoder.from_state_dict( state_dict, prefix="transcoder"               , filename=ckpt_name)
            refiner_model  =        Model.from_state_dict( state_dict, prefix="refiner.diffusion_model")
            refiner_clip   =         CLIP.from_state_dict( state_dict, prefix="refiner.conditioner"      , clip_type="stable_diffusion")
            vae            =          VAE.from_state_dict( state_dict, prefix="first_stage_model"        , filename=ckpt_name)
            upscale_model  = UpscaleModel.from_state_dict( state_dict, prefix="upscale_model")
            upscale_vae    =          VAE.from_state_dict( state_dict, prefix="refiner.first_stage_model", filename=ckpt_name)
            if upscale_model is None:
                state_dict = UPSCALE_MODELS_DIR.load_state_dict_or_raise(upscaler_name)
                upscale_model = UpscaleModel.from_state_dict( state_dict )
            return (model, None, vae, transcoder, refiner_model, refiner_clip, upscale_model, upscale_vae, metadata)


    #__ internal functions ________________________________

    @staticmethod
    def ckpt_names():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()

    @staticmethod
    def upscalers():
        return UPSCALE_MODELS_DIR.get_filename_list()

    @staticmethod
    def get_model_type(state_dict):

        # if the model contains a key that starts with `base.diffusion_model`
        # then it is a model of type "prototype1"
        for key in state_dict.keys():
            if key.startswith("refiner.first_stage_model."):
                return "TinyBreaker.prototype1"

        # otherwise, it is a model of type "prototype0" (old model)
        return "TinyBreaker.prototype0"

