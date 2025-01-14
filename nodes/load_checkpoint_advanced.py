"""
File    : load_checkpoint_advanced.py
Purpose : Node to load TinyBreaker checkpoints with customized parameters.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 7, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.utils
from .xcomfy.model       import Model
from .xcomfy.clip        import CLIP
from .xcomfy.transcoder  import Transcoder
from .xcomfy.vae         import VAE
from .utils.directories  import PIXART_CHECKPOINTS_DIR, CHECKPOINTS_DIR, VAE_DIR

_PIXART_TYPES = [
    "sigma-512",
    "sigma-1024",
    "sigma-2K"
]
_DEFAULT_PIXART_TYPE = "sigma-1024"


class LoadCheckpointAdvanced:
    TITLE       = "💪TB | Load Checkpoint Advanced"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load TinyBreaker checkpoints with customized parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name"  : (cls.model_list()      , {"tooltip": "The TinyBreaker checkpoint to load. (PixArt sigma checkpoints are also supported)"}),
                "vae"        : (cls.vae_list()        , {"tooltip": "The VAE quality to use. (Leave empty for no VAE)"}),
                "transcoder" : (cls.transcoder_list() , {"tooltip": "The TinyBreaker checkpoint to load. (PixArt sigma checkpoints are also supported)"}),
                "refiner"    : (cls.refiner_list()    , {"tooltip": "The refiner checkpoint to load. (use 'default' for auto-selection of best alternative or 'none' for no refiner)"}),
                "pixart_type": (_PIXART_TYPES         , {"tooltip": "The type and resolution of the core PixArt model in the TinyBreaker checkpoint.",
                                                         "default": _DEFAULT_PIXART_TYPE}),
                },
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "TRANSCODER", "MODEL"        , "CLIP"        , "STRING"  )
    RETURN_NAMES = ("MODEL", "VAE", "TRANSCODER", "REFINER_MODEL", "REFINER_CLIP", "METADATA")

    def load_checkpoint(self, ckpt_name, vae, transcoder, refiner, pixart_type) -> tuple:
        use_sdxl_refiner = False

        ckpt_path  = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        metadata          = None  # Metadata.from_predefined("sigma", 2048)

        model             = Model.from_state_dict(state_dict, prefix="base.diffusion_model", resolution=1024)
        vae_obj           = self.vae_object(vae, state_dict)
        transcoder_obj    = self.transcoder_object(transcoder, ckpt_name, state_dict, use_sdxl_refiner)
        refiner_model_obj = self.refiner_object(refiner, state_dict)
        refiner_clip_obj  = self.refiner_clip_object(refiner, state_dict)
        return (model, vae_obj, transcoder_obj, refiner_model_obj, refiner_clip_obj, metadata)


    #__ internal functions ________________________________

    @staticmethod
    def model_list():
        return PIXART_CHECKPOINTS_DIR.get_filename_list()


    @staticmethod
    def vae_list() -> list:
        return ["default", "high quality", *VAE_DIR.get_filename_list()]

    @staticmethod
    def vae_object(name: str, state_dict: dict) -> dict:
        if name == "default":
            return VAE.from_state_dict(state_dict, prefix="first_stage_model")
        elif name == "high quality":
            return VAE.from_state_dict(state_dict, prefix="first_stage_hqmodel")
        else:
            vae_path   = VAE_DIR.get_full_path(name)
            state_dict = comfy.utils.load_torch_file(vae_path)
            return VAE.from_state_dict(state_dict)


    @staticmethod
    def transcoder_list() -> list:
        return ["default", *VAE_DIR.get_filename_list()]

    @staticmethod
    def transcoder_object(transcoder_name   : str,
                          default_ckpt_name : str,
                          default_state_dict: dict,
                          use_sdxl_refiner  : bool = False
                          ) -> Transcoder:
        #if use_sdxl_refiner:
        #    return Transcoder.identity()

        # use default state dict
        if transcoder_name == "default":
            return Transcoder.from_state_dict(default_state_dict, prefix="transcoder", filename=default_ckpt_name)

        # load transcoder from file
        transcoder_path = VAE_DIR.get_full_path(transcoder_name)
        state_dict      = comfy.utils.load_torch_file(transcoder_path)
        return Transcoder.from_state_dict(state_dict, filename=transcoder_name)


    @staticmethod
    def refiner_list():
        return ["default", "none", *CHECKPOINTS_DIR.get_filename_list()]

    @staticmethod
    def refiner_object(name: str, state_dict: dict) -> Model:
        if name == "default":
            return Model.from_state_dict(state_dict, prefix="refiner.diffusion_model")
        else:
            refiner_path = CHECKPOINTS_DIR.get_full_path(name)
            state_dict   = comfy.utils.load_torch_file(refiner_path)
            return Model.from_state_dict(state_dict)

    @staticmethod
    def refiner_clip_object(name: str, state_dict: dict) -> CLIP:
        if name == "default":
            return CLIP.from_state_dict(state_dict, prefix="refiner.conditioner", type="stable_diffusion")
        else:
            refiner_path = CHECKPOINTS_DIR.get_full_path(name)
            state_dict   = comfy.utils.load_torch_file(refiner_path)
            return CLIP.from_state_dict(state_dict, type="stable_diffusion")


