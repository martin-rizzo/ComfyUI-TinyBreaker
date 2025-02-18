"""
File    : load_tinybreaker_checkpoint_custom.py
Purpose : Node to load TinyBreaker checkpoints with customized parameters.
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
from .xcomfy.model       import Model
from .xcomfy.vae         import VAE
from .xcomfy.clip        import CLIP
from .xcomfy.transcoder  import Transcoder
from .core.genparams     import GenParams
from .utils.directories  import TINYBREAKER_CHECKPOINTS_DIR, \
                                CHECKPOINTS_DIR, \
                                TRANSCODERS_DIR, \
                                VAE_DIR

_RESOLUTIONS = ["automatic", "512", "1024", "2K", "4K" ]
_VAE_OPTIONS = ["fast", "high quality"]
_DEFAULT_RESOLUTION = "automatic"
_DEFAULT_VAE_OPTION = "fast"


class LoadTinyBreakerCheckpointCustom:
    TITLE       = "💪TB | Load TinyBreaker Checkpoint (custom)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints with customized parameters."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name" : (cls.checkpoint_list(), {"tooltip": "The TinyBreaker checkpoint to load."}),
                "vae"       : (cls.vae_list()       , {"tooltip": "The VAE model used for encoding and decoding images to and from latent space.",
                                                       "default": _DEFAULT_VAE_OPTION}),
                "transcoder": (cls.transcoder_list(), {"tooltip": 'The transcoder model used for converting latent images from base to refiner. (use "automatic" for auto-selection of best alternative)'}),
                "refiner"   : (cls.refiner_list()   , {"tooltip": 'The refiner checkpoint to load. (use "automatic" for auto-selection of best alternative or "none" for no refiner)'}),
                "resolution": (_RESOLUTIONS         , {"tooltip": 'The base resolution the model is intended to work at. (use "automatic" for auto-selection of best alternative)',
                                                       "default": _DEFAULT_RESOLUTION}),
                },
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

    def load_checkpoint(self, ckpt_name, vae, transcoder, refiner, resolution) -> tuple:
        use_sdxl_refiner = False
        resolution = 1024

        ckpt_path  = TINYBREAKER_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        genparams      = GenParams.from_safetensors_metadata(ckpt_path)
        model_obj      = Model.from_state_dict(state_dict, prefix="base.diffusion_model", resolution=resolution)
        vae_obj        = self.vae_object(vae, ckpt_name, state_dict)
        transcoder_obj = self.transcoder_object(transcoder, ckpt_name, state_dict, use_sdxl_refiner)
        refiner_model  = self.refiner_object(refiner, state_dict)
        refiner_clip   = self.refiner_clip_object(refiner, state_dict)
        return (model_obj, vae_obj, transcoder_obj, refiner_model, refiner_clip, genparams)


    #__ internal functions ________________________________

    @staticmethod
    def checkpoint_list():
        return TINYBREAKER_CHECKPOINTS_DIR.get_filename_list()


    @staticmethod
    def vae_list() -> list:
        return [ *_VAE_OPTIONS, *VAE_DIR.get_filename_list() ]

    @staticmethod
    def vae_object(vae_name: str,
                   default_ckpt_name,
                   default_state_dict: dict
                   ) -> VAE:

        # use the fast vae stored in the default checkpoint
        if vae_name == "fast":
            return VAE.from_state_dict(default_state_dict, prefix="first_stage_model", filename=default_ckpt_name)

        # use the high quality vae stored in the default checkpoint
        if vae_name == "high quality":
            return VAE.from_state_dict(default_state_dict, prefix="first_stage_hqmodel", filename=default_ckpt_name)

        # load vae from file
        vae_path   = VAE_DIR.get_full_path(vae_name)
        state_dict = comfy.utils.load_torch_file(vae_path)
        return VAE.from_state_dict(state_dict, filename=vae_name)


    @staticmethod
    def transcoder_list() -> list:
        return ["automatic", *TRANSCODERS_DIR.get_filename_list()]

    @staticmethod
    def transcoder_object(transcoder_name   : str,
                          default_ckpt_name : str,
                          default_state_dict: dict,
                          use_sdxl_refiner  : bool = False
                          ) -> Transcoder:
        #if use_sdxl_refiner:
        #    return Transcoder.identity()

        # use the transcoder stored in the default checkpoint
        if transcoder_name == "automatic":
            return Transcoder.from_state_dict(default_state_dict, prefix="transcoder", filename=default_ckpt_name)

        # load transcoder from file
        transcoder_path = TRANSCODERS_DIR.get_full_path(transcoder_name)
        state_dict      = comfy.utils.load_torch_file(transcoder_path)
        return Transcoder.from_state_dict(state_dict, filename=transcoder_name)


    @staticmethod
    def refiner_list():
        return ["automatic", "none", *CHECKPOINTS_DIR.get_filename_list()]

    @staticmethod
    def refiner_object(name: str, state_dict: dict) -> Model:
        if name == "automatic":
            return Model.from_state_dict(state_dict, prefix="refiner.diffusion_model")
        elif name == "none":
            return None
        else:
            refiner_path = CHECKPOINTS_DIR.get_full_path(name)
            state_dict   = comfy.utils.load_torch_file(refiner_path)
            return Model.from_state_dict(state_dict)

    @staticmethod
    def refiner_clip_object(name: str, state_dict: dict) -> CLIP:
        if name == "automatic":
            return CLIP.from_state_dict(state_dict, prefix="refiner.conditioner", clip_type="stable_diffusion")
        elif name == "none":
            return None
        else:
            refiner_path = CHECKPOINTS_DIR.get_full_path(name)
            state_dict   = comfy.utils.load_torch_file(refiner_path)
            return CLIP.from_state_dict(state_dict, clip_type="stable_diffusion")


