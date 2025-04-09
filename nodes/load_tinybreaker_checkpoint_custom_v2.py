"""
File    : load_tinybreaker_checkpoint_custom_v2.py
Purpose : Node to load TinyBreaker checkpoints with customization parameters.
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
import comfy.utils
from .xcomfy.model        import Model
from .xcomfy.vae          import VAE
from .xcomfy.clip         import CLIP
from .xcomfy.transcoder   import Transcoder
from .functions.genparams import GenParams
from .utils.directories   import TINYBREAKER_CHECKPOINTS_DIR, \
                                 CHECKPOINTS_DIR, \
                                 TRANSCODERS_DIR, \
                                 VAE_DIR


_AUTOMATIC   = "auto"
_EMBEDDED    = "embedded"
_VAE_FAST    = "fast"
_VAE_QUALITY = "high quality"
_VAE_TYPES   = [ _AUTOMATIC, _VAE_QUALITY ]
_RESOLUTIONS = [ _AUTOMATIC, "512", "1024", "2K", "4K" ]


class LoadTinyBreakerCheckpointCustomV2:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint (custom)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load a TinyBreaker checkpoints with customization parameters."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "ckpt_name" : (cls.ckpt_names() , {"tooltip": "The TinyBreaker checkpoint to load."
                                              }),
            "vae"       : (cls.vaes()       , {"tooltip": "The VAE model used for encoding and decoding images to and from latent space.",
                                               "default": _AUTOMATIC,
                                              }),
            "transcoder": (cls.transcoders(), {"tooltip": 'The transcoder model used for converting latent images from base to refiner. (use "automatic" for auto-selection of best alternative)',
                                               "default": _AUTOMATIC,
                                              }),
            "refiner"   : (cls.refiners()   , {"tooltip": 'The refiner checkpoint to load. (use "automatic" for auto-selection of best alternative or "none" for no refiner)',
                                               "default": _AUTOMATIC,
                                              }),
            "resolution": (_RESOLUTIONS     , {"tooltip": 'The base resolution the model is intended to work at. (use "automatic" for auto-selection of best alternative)',
                                               "default": _AUTOMATIC,
                                              }),
            },
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

    def load_checkpoint(self, ckpt_name, vae, transcoder, refiner, resolution) -> tuple:
        vae_name        = vae
        transcoder_name = transcoder if refiner != "none" else ""
        refiner_name    = refiner    if refiner != "none" else ""

        # resolve the automatic settings
        if vae_name        == _AUTOMATIC: vae_name        = _VAE_FAST
        if transcoder_name == _AUTOMATIC: transcoder_name = _EMBEDDED
        if refiner_name    == _AUTOMATIC: refiner_name    = _EMBEDDED

        use_sdxl_refiner = False
        resolution = 1024

        ckpt_full_path = TINYBREAKER_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict     = TINYBREAKER_CHECKPOINTS_DIR.load_state_dict_or_raise(ckpt_name)

        genparams     = GenParams.from_safetensors_metadata(ckpt_full_path)
        model         = Model.from_state_dict( state_dict, prefix="base.diffusion_model", resolution=resolution )
        vae           = self.vae_object(          vae_name       , ckpt_name, state_dict                   )
        transcoder    = self.transcoder_object(   transcoder_name, ckpt_name, state_dict, use_sdxl_refiner )
        refiner_model = self.refiner_object(      refiner_name   , ckpt_name, state_dict                   )
        refiner_clip  = self.refiner_clip_object( refiner_name   , ckpt_name, state_dict                   )

        return (model, None, transcoder, refiner_model, refiner_clip, vae, genparams)


    #__ internal functions ________________________________

    @staticmethod
    def ckpt_names():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()

    @staticmethod
    def vaes() -> list:
        return [ *_VAE_TYPES, *VAE_DIR.get_filename_list() ]

    @staticmethod
    def transcoders() -> list:
        return [ _AUTOMATIC, *TRANSCODERS_DIR.get_filename_list()]

    @staticmethod
    def refiners():
        return [ _AUTOMATIC, "none", *CHECKPOINTS_DIR.get_filename_list()]


    @staticmethod
    def vae_object(vae_name       : str,
                   main_ckpt_name : str,
                   main_state_dict: dict,
                   ) -> VAE:
        assert vae_name, "vae_name must be set"

        # use the fast vae that is embedded in the main checkpoint
        if vae_name == _VAE_FAST:
            return VAE.from_state_dict(main_state_dict, prefix="first_stage_model", filename=main_ckpt_name)

        # use the high quality vae that is embedded in the main checkpoint
        if vae_name == _VAE_QUALITY:
            return VAE.from_state_dict(main_state_dict, prefix="first_stage_hqmodel", filename=main_ckpt_name)

        # load vae from file
        state_dict = VAE_DIR.load_state_dict_or_raise(vae_name)
        return VAE.from_state_dict(state_dict, prefix="", filename=vae_name)


    @staticmethod
    def transcoder_object(transcoder_name : str,
                          main_ckpt_name  : str,
                          main_state_dict : dict,
                          use_sdxl_refiner: bool,
                          ) -> Transcoder | None:
        if not transcoder_name:
            return None

        #if use_sdxl_refiner:
        #    return Transcoder.identity()

        # use the transcoder that is embedded in the main checkpoint
        if transcoder_name == _EMBEDDED:
            return Transcoder.from_state_dict(main_state_dict, prefix="transcoder", filename=main_ckpt_name)

        # load transcoder from file
        state_dict = TRANSCODERS_DIR.load_state_dict_or_raise(transcoder_name)
        return Transcoder.from_state_dict(state_dict, prefix="", filename=transcoder_name)


    @staticmethod
    def refiner_object(refiner_name   : str,
                       main_ckpt_name : str,
                       main_state_dict: dict
                       ) -> Model | None:
        if not refiner_name:
            return None

        # use the refiner that is embedded in the main checkpoint
        if refiner_name == _EMBEDDED:
            return Model.from_state_dict(main_state_dict, prefix="refiner.diffusion_model")

        # load the refiner from file
        state_dict = CHECKPOINTS_DIR.load_state_dict_or_raise(refiner_name)
        return Model.from_state_dict(state_dict, prefix="")


    @staticmethod
    def refiner_clip_object(refiner_name   : str,
                            main_ckpt_name : str,
                            main_state_dict: dict
                            ) -> CLIP | None:
        if not refiner_name:
            return None

        # use the refiner CLIP that is embedded in the main checkpoint
        if refiner_name == _EMBEDDED:
            return CLIP.from_state_dict(main_state_dict, prefix="refiner.conditioner", clip_type="stable_diffusion")

        # load the CLIP that is embedded in the refiner checkpoint
        # TODO: verificar que el prefijo "*" detecte correctamente el clip en SD/SDXL
        state_dict = CHECKPOINTS_DIR.load_state_dict_or_raise(refiner_name)
        return CLIP.from_state_dict(state_dict, prefix="*", clip_type="stable_diffusion")


