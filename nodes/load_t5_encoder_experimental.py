"""
File    : load_t5_encoder_experimental.py
Purpose : Node to load the T5 encoder using experimental methods
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 17, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.text_encoders.t5
from .xcomfy.clip               import CLIP
from .utils.directories         import TEXT_ENCODERS_DIR
from .core.t5_encoder_model_cmf import T5EncoderModelCmf


class LoadT5EncoderExperimental:
    TITLE       = "💪TB | Load T5 Encoder (Experimental)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Load a T5 encoder using experimental methods to utilize limited GPU memory on mid-range or low-end GPUs."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "t5_name": (cls.t5_name_list(), {"tooltip": "Name of the T5 encoder checkpoint to load."}),
            "mode"   : (cls.modes()       , {"tooltip": "Select the desired method for loading the T5 encoder:\n\n* \"comfyui native\": Utilizes ComfyUI's built-in functionality for loading and handling the T5 encoder.\n\n* \"experimental\": Employs experimental techniques designed to optimize memory usage, particularly beneficial for mid-range or low-end GPUs."})
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_t5_checkpoint"
    RETURN_TYPES    = ("CLIP",)
    OUTPUT_TOOLTIPS = ("The loaded T5 Encoder ready for use as a CLIP connection.",)

    def load_t5_checkpoint(self, t5_name, mode="experimental"):
        # model_options = {}
        # if device == "cpu":
        #     model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        t5_encoder_class = comfy.text_encoders.t5.T5
        if mode == "experimental":
            t5_encoder_class = T5EncoderModelCmf

        state_dict       = TEXT_ENCODERS_DIR.load_state_dict_or_raise(t5_name)
        clip = CLIP.from_custom_t5_encoder(t5_encoder_class, state_dict, prefix="", clip_type="sd3")
        return (clip,)


    #__ internal functions ________________________________

    @staticmethod
    def t5_name_list():
        return TEXT_ENCODERS_DIR.get_filename_list()

    @staticmethod
    def modes():
        return ("comfyui native", "experimental")


