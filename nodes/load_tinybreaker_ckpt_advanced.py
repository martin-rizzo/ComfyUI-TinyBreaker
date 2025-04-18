"""
File    : load_tinybreaker_ckpt_advanced.py
Purpose : Node to load TinyBreaker checkpoints with advanced settings.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 7, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams                  import GenParams
from .core.comfyui_bridge.model       import Model
from .core.comfyui_bridge.vae         import VAE
from .core.comfyui_bridge.clip        import CLIP
from .core.comfyui_bridge.transcoder  import Transcoder
from .core.directories  import TINYBREAKER_CHECKPOINTS_DIR, \
                               CHECKPOINTS_DIR, \
                               TRANSCODERS_DIR, \
                               VAE_DIR
_AUTOMATIC        = "auto"
_EMBEDDED         = "embedded"
_NONE             = "none"
_VAE_TYPE_FAST    = "fast"
_VAE_TYPE_QUALITY = "high_quality"
_VAE_TYPES   = [ _AUTOMATIC, _VAE_TYPE_FAST, _VAE_TYPE_QUALITY ]
_TRANSCODERS = [ _AUTOMATIC, _EMBEDDED ]
_REFINERS    = [ _AUTOMATIC, _EMBEDDED ] # _NONE
_RESOLUTIONS = [ _AUTOMATIC, "512", "1024", "2048", "4096" ]


class LoadTinyBreakerCkptAdvanced:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint (advanced)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Loads a TinyBreaker checkpoint allowing for detailed configuration of each sub-component."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "ckpt_name" : (cls.ckpt_names() ,{"tooltip": "The TinyBreaker checkpoint to load."
                                             }),
            "vae"       : (cls.vaes()       ,{"tooltip": "The VAE model used for encoding and decoding images to and from latent space.",
                                              "default": _AUTOMATIC,
                                             }),
            "transcoder": (cls.transcoders(),{"tooltip": 'The transcoder model used for converting latent images from base to refiner. (use "automatic" for auto-selection of best alternative)',
                                              "default": _AUTOMATIC,
                                             }),
            "refiner"   : (cls.refiners()   ,{"tooltip": 'The refiner checkpoint to load. (use "automatic" for auto-selection of best alternative or "none" for no refiner)',
                                              "default": _AUTOMATIC,
                                             }),
            "resolution": (_RESOLUTIONS     ,{"tooltip": 'The base resolution the model is intended to work at. (use "automatic" for auto-selection of best alternative)',
                                              "default": _AUTOMATIC,
                                             }),
            "upscaler_vae": (cls.vaes()     ,{"tooltip": "The VAE used during upscaling. A `high_quality` VAE is available but due to its high VRAM consumption, `auto` and `fast` are recommended.",
                                              "default": _AUTOMATIC
                                             }),
            },
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

    def load_checkpoint(self, ckpt_name, vae, transcoder, refiner, resolution, upscaler_vae):
        vae_name          = vae
        refiner_name      = refiner
        transcoder_name   = transcoder if refiner != _NONE else _NONE
        upscaler_vae_name = upscaler_vae

        # resolve the automatic settings
        vae_name          = _VAE_TYPE_FAST if vae_name          == _AUTOMATIC else vae_name
        refiner_name      = _EMBEDDED      if refiner_name      == _AUTOMATIC else refiner_name
        transcoder_name   = _EMBEDDED      if transcoder_name   == _AUTOMATIC else transcoder_name
        upscaler_vae_name = _VAE_TYPE_FAST if upscaler_vae_name == _AUTOMATIC else upscaler_vae_name
        resolution        = 1024           if resolution        == _AUTOMATIC else int(resolution)
        sdxl_refiner      = False  # SDXL refiners are not supported yet

        # load the main model checkpoint
        ckpt_full_path = TINYBREAKER_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict     = TINYBREAKER_CHECKPOINTS_DIR.load_state_dict_or_raise(ckpt_name)
        metadata       = GenParams.from_safetensors_metadata(ckpt_full_path)

        model         = Model.from_state_dict( state_dict, prefix="base.diffusion_model", resolution=resolution )
        transcoder    = self.transcoder_obj  ( transcoder_name  , ckpt_name, state_dict, prefix="transcoder"               , sdxl_refiner=sdxl_refiner   )
        refiner_model = self.refiner_obj     ( refiner_name     , ckpt_name, state_dict, prefix="refiner.diffusion_model"  )
        refiner_clip  = self.refiner_clip_obj( refiner_name     , ckpt_name, state_dict, prefix="refiner.conditioner"      , clip_type="stable_diffusion")
        vae           = self.vae_obj         ( vae_name         , ckpt_name, state_dict, prefix="first_stage_model"        )
        upscaler_vae  = self.vae_obj         ( upscaler_vae_name, ckpt_name, state_dict, prefix="refiner.first_stage_model")
        return (model, None, vae, transcoder, refiner_model, refiner_clip, upscaler_vae, metadata)


    #__ internal functions ________________________________

    @staticmethod
    def ckpt_names():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()

    @staticmethod
    def vaes() -> list:
        return [ *_VAE_TYPES, *VAE_DIR.get_filename_list() ]

    @staticmethod
    def transcoders() -> list:
        return [ *_TRANSCODERS, *TRANSCODERS_DIR.get_filename_list()]

    @staticmethod
    def refiners():
        return [ *_REFINERS, *CHECKPOINTS_DIR.get_filename_list()]


    @staticmethod
    def vae_obj(vae_name  : str,
                ckpt_name : str,
                state_dict: dict,
                /,*,
                prefix    : str,
                ) -> VAE:

        # use the transcoder that is embedded in the main checkpoint
        if vae_name == _EMBEDDED or vae_name == _VAE_TYPE_FAST or vae_name == _VAE_TYPE_QUALITY:
            return VAE.from_state_dict(state_dict, prefix=prefix, filename=ckpt_name)

        # load vae from file
        state_dict = VAE_DIR.load_state_dict_or_raise(vae_name)
        return VAE.from_state_dict(state_dict, prefix="", filename=vae_name)


    @staticmethod
    def transcoder_obj(transcoder_name : str,
                       ckpt_name       : str,
                       state_dict      : dict,
                       /,*,
                       prefix          : str,
                       sdxl_refiner    : bool,
                       ) -> Transcoder | None:

        # transcoder can be disabled by setting refiner to "none"
        if transcoder_name == _NONE:
            return None

        #if use_sdxl_refiner:
        #    return Transcoder.identity()

        # use the transcoder that is embedded in the main checkpoint
        if transcoder_name == _EMBEDDED:
            return Transcoder.from_state_dict(state_dict, prefix=prefix, filename=ckpt_name)

        # load transcoder from file
        state_dict = TRANSCODERS_DIR.load_state_dict_or_raise(transcoder_name)
        return Transcoder.from_state_dict(state_dict, prefix="", filename=transcoder_name)


    @staticmethod
    def refiner_obj(refiner_name: str,
                    ckpt_name   : str,
                    state_dict  : dict,
                    /,*,
                    prefix      : str
                    ) -> Model | None:

        # refiner can be disabled by setting it to "none"
        if refiner_name == _NONE:
            return None

        # use the refiner that is embedded in the main checkpoint
        if refiner_name == _EMBEDDED:
            return Model.from_state_dict(state_dict, prefix=prefix)

        # load the refiner from file
        state_dict = CHECKPOINTS_DIR.load_state_dict_or_raise(refiner_name)
        return Model.from_state_dict(state_dict, prefix="")


    @staticmethod
    def refiner_clip_obj(refiner_clip_name: str,
                         ckpt_name        : str,
                         state_dict       : dict,
                         /,*,
                         prefix          : str,
                         clip_type       : str
                         ) -> CLIP | None:

        # refiner can be disabled by setting it to "none"
        if refiner_clip_name == _NONE:
            return None

        # use the refiner CLIP that is embedded in the main checkpoint
        if refiner_clip_name == _EMBEDDED:
            return CLIP.from_state_dict(state_dict, prefix=prefix, clip_type=clip_type)

        # load the CLIP that is embedded in the refiner checkpoint
        # TODO: verificar que el prefijo "*" detecte correctamente el clip en SD/SDXL
        state_dict = CHECKPOINTS_DIR.load_state_dict_or_raise(refiner_clip_name)
        return CLIP.from_state_dict(state_dict, prefix="*", clip_type=clip_type)


