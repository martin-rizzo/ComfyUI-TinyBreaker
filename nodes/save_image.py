"""
File    : save_image.py
Purpose : Node for saving a generated images to disk including A1111/CivitAI embedded metadata.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import time
import json
import numpy as np
import folder_paths
from PIL                import Image
from PIL.PngImagePlugin import PngInfo
from .core.genparams    import GenParams
from ._common            import ireplace

_A1111_SAMPLER_BY_COMFY_NAME = {
    "euler"                    : "Euler",
    "euler_cfg_pp"             : "Euler",
    "euler_ancestral"          : "Euler a",
    "euler_ancestral_cfg_pp"   : "Euler a",
    "heun"                     : "Heun",
    "heunpp2"                  : "Heun",
    "dpm_2"                    : "DPM2",
    "dpm_2_ancestral"          : "DPM2 a",
    "lms"                      : "LMS",
    "dpm_fast"                 : "DPM fast",
    "dpm_adaptive"             : "DPM adaptive",
    "dpmpp_2s_ancestral"       : "DPM++ 2S a",
    "dpmpp_2s_ancestral_cfg_pp": "DPM++ 2S a",
    "dpmpp_sde"                : "DPM++ SDE",
    "dpmpp_sde_gpu"            : "DPM++ SDE",
    "dpmpp_2m"                 : "DPM++ 2M",
    "dpmpp_2m_cfg_pp"          : "DPM++ 2M",
    "dpmpp_2m_sde"             : "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu"         : "DPM++ 2M SDE",
    "dpmpp_3m_sde"             : "DPM++ 3M SDE",
    "dpmpp_3m_sde_gpu"         : "DPM++ 3M SDE",
    "lcm"                      : "LCM",
    "ddim"                     : "DDIM",
    "uni_pc"                   : "UniPC",
    "uni_pc_bh2"               : "UniPC",
# unsupported samplers
    "ddpm"                     : "!DDPM",
    "ipndm"                    : "!iPNDM",
    "ipndm_v"                  : "!iPNDM_v",
    "deis"                     : "!DEIS",
    "res_multistep"            : "!RES Multistep",
    "res_multistep_cfg_pp"     : "!RES Multistep",
}

_A1111_SCHEDULER_BY_COMFY_NAME = {
    "normal"          : "",
    "karras"          : " Karras",
    "exponential"     : " Exponential",
#    "sgm_uniform"     : "",
#    "simple"          : "",
#    "ddim_uniform"    : "",
#    "beta"            : "",
#    "linear_quadratic": "",
#    "kl_optimal"      : "",
}


class SaveImage:
    TITLE       = "💪TB | Save Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Saves generated images along with A1111/CivitAI metadata within PNG files. This facilitates easy extraction of prompts and settings through widely available tools."
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images"         : ("IMAGE" , {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                                               "default": "TinyBreaker"
                                               }),
            },
            "optional": {
                "genparams": ("GENPARAMS", {"tooltip": "An optional input with the generation parameters to embed in the image using the A1111/CivitAI format."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "save_images"
    RETURN_TYPES = ()

    def save_images(self,
                    images,
                    filename_prefix: str,
                    genparams      : GenParams = None,
                    prompt         : dict      = None,
                    extra_pnginfo  : dict      = None
                    ):

        image_width  = images[0].shape[1]
        image_height = images[0].shape[0]
        a1111_params = self._create_a1111_params(genparams, image_width, image_height)

        # create PNG info containing A1111/CivitAI+ComfyUI metadata
        pnginfo = PngInfo()
        if a1111_params:
            pnginfo.add_text("parameters", a1111_params)
        if prompt:
            pnginfo.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo:
            for key, content_dict in extra_pnginfo.items():
                if key != "parameters":
                    pnginfo.add_text(key, json.dumps(content_dict))

        # solve the `filename_prefix`` entered by the user and get the full path
        filename_prefix = \
            self._solve_filename_variables(f"{filename_prefix}{self.extra_prefix}", genparams=genparams)
        full_output_folder, filename, counter, subfolder, filename_prefix \
            = folder_paths.get_save_image_path(filename_prefix,
                                               self.output_dir,
                                               image_width,
                                               image_height
                                               )

        image_locations = []
        for batch_number, image in enumerate(images):

            # convert to PIL Image
            image = np.clip( image.numpy(force=True) * 255, 0, 255 ) # <- numpy
            image = Image.fromarray( image.astype(np.uint8) )        # <- PIL

            # generate the full file path to save the image
            filename  = filename.replace("%batch_num%", str(batch_number))
            filename  = f"{filename}_{counter+batch_number:04}_.png"
            file_path =  os.path.join(full_output_folder, filename)

            image.save(file_path,
                       pnginfo        = pnginfo,
                       compress_level = self.compress_level
                       )
            image_locations.append({"filename" : filename,
                                    "subfolder": subfolder,
                                    "type"     : self.type
                                    })

        return { "ui": { "images": image_locations } }


    def __init__(self,
                 *,# keyword-only arguments #
                 output_dir  : str = folder_paths.get_output_directory(),
                 type        : str = "output",
                 extra_prefix: str = ""
                 ):
        """
        This initializer is configurable to be able to derive a child class
        that saves images in a different directory.
        """
        self.output_dir     = output_dir
        self.type           = type
        self.extra_prefix   = extra_prefix
        self.compress_level = 4 if type == "output" else 0


    #__ internal functions ________________________________

    @staticmethod
    def _create_a1111_params(genparams   : GenParams | None,
                             image_width : int,
                             image_height: int
                             ) -> str:
        """
        Return a string containing generation parameters in A1111 format.
        Args:
            genparams   : A GenParams dictionary containing all the generation parameters.
            image_width : The width of the generated image.
            image_height: The height of the generated image.
        """
        if not genparams:
            return ""


        def a1111_normalized_string(text: str, /) -> str:
            return text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


        def a1111_sampler_name(comfy_sampler: str, comfy_scheduler: str, support_all_samplers: bool = False, /) -> str:
            DEFAULT_SAMPLER = "Euler"
            if not comfy_sampler:
                return ""
            a1111_sampler = _A1111_SAMPLER_BY_COMFY_NAME.get(comfy_sampler, DEFAULT_SAMPLER)
            if not support_all_samplers and a1111_sampler.startswith("!"):
                a1111_sampler = DEFAULT_SAMPLER
            a1111_sampler = a1111_sampler.lstrip('!') + _A1111_SCHEDULER_BY_COMFY_NAME.get(comfy_scheduler or "normal", "")
            return a1111_sampler


        # base/refiner prefixes
        BASE = "denoising.base."
        RE__ = "denoising.refiner."

        # extract and clean up parameters from the GenParams dictionary
        positive            = a1111_normalized_string( genparams.get("user.prompt"  , "") )
        negative            = a1111_normalized_string( genparams.get("user.negative", "") )
        sampler             = a1111_sampler_name( genparams.get(f"{BASE}sampler"), genparams.get(f"{BASE}scheduler") )
        refiner_sampler     = a1111_sampler_name( genparams.get(f"{RE__}sampler"), genparams.get(f"{RE__}scheduler"), True )
        base_cfg            = genparams.get_float(f"{BASE}cfg")
        refiner_cfg         = genparams.get_float(f"{RE__}cfg", 0)
        base_steps_start    = genparams.get_int(f"{BASE}steps_start",0)
        refiner_steps_start = genparams.get_int(f"{RE__}steps_start",0)
        base_steps_end      = min( genparams.get_int(f"{BASE}steps",0), genparams.get_int(f"{BASE}steps_end",10000) )
        refiner_steps_end   = min( genparams.get_int(f"{RE__}steps",0), genparams.get_int(f"{RE__}steps_end",10000) )
        seed                = genparams.get_int(f"{BASE}noise_seed")
        width               = image_width
        height              = image_height
        base_checkpoint     = genparams.get("file.name", "TinyBreaker.safetensors")
        base_checkpoint     = os.path.splitext(base_checkpoint)[0]
        refiner_checkpoint  = None #f"{base_checkpoint}.refiner"
        base_steps          = max(0, base_steps_end-base_steps_start)
        refiner_steps       = max(0, refiner_steps_end-refiner_steps_start)
        total_steps         = base_steps + refiner_steps

        # build A1111 params string
        a1111_params = f"{positive}\nNegative prompt: {negative}\nSteps: {total_steps}, "
        if sampler:
            a1111_params += f"Sampler: {sampler}, "
        if base_cfg:
            a1111_params += f"CFG scale: {base_cfg}, "
        if seed:
            a1111_params += f"Seed: {seed}, "
        if width and height:
            a1111_params += f"Size: {image_width}x{image_height}, "
        if base_checkpoint:
            a1111_params += f"Model: {base_checkpoint}, "
        if refiner_checkpoint and refiner_sampler:
            a1111_params += f"Denoising strength: {refiner_cfg}, "
            a1111_params += f"Hires checkpoint: {refiner_checkpoint}, "
            a1111_params += f"Hires upscaler: None, "
            a1111_params += f"Hires resize: {image_width}x{image_height}, "
            a1111_params += f"Hires sampler: {refiner_sampler}, "
            a1111_params += f"Hires steps: {refiner_steps}, "
        #if discard_penultimate_sigma:
        #    a1111_params += "Discard penultimate sigma: True, "

        # remove the trailing comma and return it
        a1111_params = a1111_params.strip().rstrip(",")
        return a1111_params


    def _solve_filename_variables(self,
                                  filename : str,
                                  *,# keyword-only args #
                                  genparams: GenParams
                                  ) -> str:
        """
        Solve the filename variables and return a string containing the solved filename.
        Args:
            filename    : The filename to solve.
            genparams   : A GenParams dictionary containing all the generation parameters.
        """
        now: time.struct_time = time.localtime()

        def get_var_value(name: str) -> str | None:
                """Returns the value for a given variable name or None if the variable name is not defined."""
                case_name = name
                name      = case_name.lower()
                if name == "":
                    return "%"
                # try to resolve time variables
                elif name == "year"  : return str(now.tm_year)
                elif name == "month" : return str(now.tm_mon ).zfill(2)
                elif name == "day"   : return str(now.tm_mday).zfill(2)
                elif name == "hour"  : return str(now.tm_hour).zfill(2)
                elif name == "minute": return str(now.tm_min ).zfill(2)
                elif name == "second": return str(now.tm_sec ).zfill(2)
                # try to resolve full date variable
                elif name.startswith("date:"):
                    value = case_name[5:]
                    value = ireplace(value, "yyyy", str(now.tm_year))
                    value = ireplace(value, "yy"  , str(now.tm_year)[-2:])
                    value = value.replace(  "MM"  , str(now.tm_mon ).zfill(2))
                    value = ireplace(value, "dd"  , str(now.tm_mday).zfill(2))
                    value = ireplace(value, "hh"  , str(now.tm_hour).zfill(2))
                    value = value.replace(  "mm"  , str(now.tm_min ).zfill(2))
                    value = ireplace(value, "ss"  , str(now.tm_sec ).zfill(2))
                    return value
                elif name in genparams:
                    value = str(genparams[name])[:16]
                return None

        output = ""
        next_token_is_var = False
        for token in filename.split("%"):
            current_token_is_var = next_token_is_var
            last_token_was_text  = current_token_is_var

            # if the token contains spaces then it's not a variable name
            if ' ' in token:
                current_token_is_var = False

            var_value = get_var_value(token) if current_token_is_var else None
            if var_value is not None:
                # current token is a variable and the next token is text
                output += var_value
                next_token_is_var = False
            else:
                # current token is text, and the next token could be a variable
                output += ("%" if last_token_was_text else "") + token
                next_token_is_var = True

        return output



