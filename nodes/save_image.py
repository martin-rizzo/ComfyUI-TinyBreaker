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
from PIL                                 import Image
from PIL.PngImagePlugin                  import PngInfo
from .core.genparams                     import GenParams
from .core.comfyui_bridge.helpers.images import normalize_images
from .core.genparams_from_prompt         import split_prompt_and_args, join_prompt_and_args
from ._common                            import ireplace, get_image_size_from_genparams

_A1111_SAMPLER_BY_COMFY_SAMPLER_SCHEDULER = {
    "euler_ancestral normal"   : "Euler a",
    "euler normal"             : "Euler",
    "lms normal"               : "LMS",
    "heun normal"              : "Heun",
    "dpmpp_2 normal"           : "DPM2",
    "dpmpp_2_ancestral normal" : "DPM2 a",
    "dpmpp_2s_ancestral normal": "DPM++ 2S a",
    "dpmpp_2m normal"          : "DPM++ 2M",
    "dpmpp_2m_sde normal"      : "DPM++ 2M SDE",
    "dpmpp_sde normal"         : "DPM++ SDE",
    "dpm_fast normal"          : "DPM fast",
    "dpm_adaptive normal"      : "DPM adaptive",
    "lms karras"               : "LMS Karras",
    "dpm_2 karras"             : "DPM2 Karras",
    "dpm_2_ancestral karras"   : "DPM2 a Karras",
    "dpmpp_2s_ancestral karras": "DPM++ 2S a Karras",
    "dpmpp_2m karras"          : "DPM++ 2M Karras",
    "dpmpp_2m_sde karras"      : "DPM++ 2M SDE Karras",
    "dpmpp_sde karras"         : "DPM++ SDE Karras",
    "dpmpp_3m_sde normal"      : "DPM++ 3M SDE",
    "dpmpp_3m_sde karras"      : "DPM++ 3M SDE Karras",
    "dpmpp_3m_sde exponential" : "DPM++ 3M SDE Exponential",
    "ddim normal"              : "DDIM",
    "plms normal"              : "PLMS",
    "uni_pc normal"            : "UniPC",
    "lcm normal"               : "LCM",
}

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
    "ddpm"                          : "!DDPM",
    "ipndm"                         : "!iPNDM",
    "ipndm_v"                       : "!iPNDM_v",
    "deis"                          : "!DEIS",
    "res_multistep"                 : "!RES Multistep",
    "res_multistep_cfg_pp"          : "!RES Multistep",
    "res_multistep_ancestral"       : "!RES Multistep a",
    "res_multistep_ancestral_cfg_pp": "!RES Multistep a",
    "gradient_estimation"           : "!Gradient Estimation",
    "er_sde"                        : "!ER SDE",
    "seeds_2"                       : "!Seeds 2",
    "seeds_3"                       : "!Seeds 3",
}

_A1111_SCHEDULER_BY_COMFY_NAME = {
    "normal"      : "Normal",
    "karras"      : "Karras",
    "exponential" : "Exponential",
    "sgm_uniform" : "SGM Uniform",
    "simple"      : "Simple",
    "ddim_uniform": "Uniform",
    "beta"        : "Beta",
    "kl_optimal"  : "KL Optimal",
    # unsupported schedulers
    "linear_quadratic": "!Linear Quadratic",
}


class SaveImage:
    TITLE       = "💪TB | Save Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Saves generated images along with A1111/CivitAI metadata within PNG files. This facilitates easy extraction of prompts and settings through widely available tools."
    OUTPUT_NODE = True

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "images"         : ("IMAGE" , {"tooltip": "The images to save."
                                          }),
            "filename_prefix": ("STRING", {"tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                                           "default": "TinyBreaker"
                                          }),
        },
        "optional": {
            "genparams": ("GENPARAMS", {"tooltip": "An optional input with the generation parameters to embed in the image using the A1111/CivitAI format."
                                       }),
        },
        "hidden": {
            "prompt"       : "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO"
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
        images = normalize_images(images)
        image_width  = images[0].shape[1]
        image_height = images[0].shape[0]
        noise_seed   = genparams.get_int("denoising.base.noise_seed", None)
        workflow     = extra_pnginfo.get("workflow") if extra_pnginfo else None

        # update random seed in the unified prompt
        # (eg: replace "--seed random" by "--seed 234567")
        prompt    = _update_unified_prompt(prompt,    random_seed=noise_seed)
        workflow  = _update_unified_prompt(workflow,  random_seed=noise_seed)
        genparams = _update_unified_prompt(genparams, random_seed=noise_seed)

        # create PNG info containing A1111/CivitAI+ComfyUI metadata
        pnginfo = PngInfo()

        if genparams:
            a1111_parameters = self._create_a1111_params(genparams, output_image_size=(image_width, image_height))
            pnginfo.add_text("parameters", a1111_parameters)

        if prompt:
            prompt_json = json.dumps(prompt)
            pnginfo.add_text("prompt", prompt_json)

        if workflow:
            workflow_json = json.dumps(workflow)
            pnginfo.add_text("workflow", workflow_json)

        if extra_pnginfo:
            for info_name, info_dict in extra_pnginfo.items():
                if info_name not in ("parameters", "prompt", "workflow"):
                    pnginfo.add_text(info_name, json.dumps(info_dict))


        # solve the `filename_prefix`` entered by the user and get the full path
        filename_prefix = \
            self._solve_filename_variables(f"{filename_prefix}{self.extra_prefix}", genparams=genparams)
        full_output_folder, name, counter, subfolder, filename_prefix \
            = folder_paths.get_save_image_path(filename_prefix,
                                               self.output_dir,
                                               image_width,
                                               image_height
                                               )

        image_locations = []
        for batch_number, image in enumerate(images):
            batch_name = name.replace("%batch_num%", str(batch_number))

            # convert to PIL Image
            image = np.clip( image.numpy(force=True) * 255, 0, 255 ) # <- numpy
            image = Image.fromarray( image.astype(np.uint8) )        # <- PIL

            # generate the full file path to save the image
            filename  = f"{batch_name}_{counter+batch_number:04}_.png"
            file_path =  os.path.join(full_output_folder, filename)

            image.save(file_path,
                       pnginfo        = pnginfo,
                       compress_level = self.compress_level)
            image_locations.append({"filename" : filename,
                                    "subfolder": subfolder,
                                    "type"     : self.type})

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
                             output_image_size: tuple[int, int],
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

        # base/refiner prefixes
        BASE = "denoising.base."
        RE__ = "denoising.refiner."

        # extract and clean up parameters from the GenParams dictionary
        width, height        = get_image_size_from_genparams(genparams)
        positive             = genparams.get_str("user.prompt")
        negative             = genparams.get_str("user.negative")
        style                = genparams.get_str("user.style")
        comfy_base_sampler   = genparams.get_str(f"{BASE}sampler")
        comfy_base_scheduler = genparams.get_str(f"{BASE}scheduler")
        comfy_re_sampler     = genparams.get_str(f"{RE__}sampler")
        comfy_re_scheduler   = genparams.get_str(f"{RE__}scheduler")
        sampler              = _a1111_sampler_name(comfy_base_sampler, comfy_base_scheduler, include_unsupported=True)
        refiner_sampler      = _a1111_sampler_name(comfy_re_sampler  , comfy_re_scheduler  , include_unsupported=True)
        schedule_type        = None
        base_cfg             = genparams.get_float(f"{BASE}cfg")
        refiner_cfg          = genparams.get_float(f"{RE__}cfg", 0)
        base_steps           = genparams.get_int(f"{BASE}steps",0)
        refiner_steps        = genparams.get_int(f"{RE__}steps",0)
        base_steps_start     = genparams.get_int(f"{BASE}steps_start",0)
        refiner_steps_start  = genparams.get_int(f"{RE__}steps_start",0)
        base_steps_end       = min( genparams.get_int(f"{BASE}steps",0), genparams.get_int(f"{BASE}steps_end",10000) )
        refiner_steps_end    = min( genparams.get_int(f"{RE__}steps",0), genparams.get_int(f"{RE__}steps_end",10000) )
        base_seed            = genparams.get_int(f"{BASE}noise_seed")
        refiner_seed         = genparams.get_int(f"{RE__}noise_seed")
        base_checkpoint      = genparams.get("file.name", "TinyBreaker.safetensors")
        base_checkpoint      = os.path.splitext(base_checkpoint)[0]
        refiner_checkpoint   = None #f"{base_checkpoint}.refiner"
        total_steps          = max(0, base_steps_end-base_steps_start) + max(0, refiner_steps_end-refiner_steps_start)
        discard_penul_sigma  = False
        tb_base_params       = _a1111_compact_params(comfy_base_sampler, comfy_base_scheduler, base_steps   , base_steps_start   , base_steps_end   , base_cfg   , base_seed   )
        tb_refiner_params    = _a1111_compact_params(comfy_re_sampler  , comfy_re_scheduler  , refiner_steps, refiner_steps_start, refiner_steps_end, refiner_cfg, refiner_seed)

        # set up the A1111 compatible parameters
        #   name of standard parameters used by A1111 can be found in:
        #     - https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.10.1/modules/processing.py#L770
        #   name of extra extended parameters can be found
        #     searching for "extra_generation_params" in the A1111 repository.
        params = { "Steps": total_steps }
        if sampler             : params["Sampler"                  ] = sampler
        if schedule_type       : params["Schedule type"            ] = schedule_type
        if base_cfg            : params["CFG scale"                ] = base_cfg
        if base_seed           : params["Seed"                     ] = base_seed
        if width and height    : params["Size"                     ] = f"{width}x{height}"
        if base_checkpoint     : params["Model"                    ] = base_checkpoint
        if discard_penul_sigma : params["Discard penultimate sigma"] = "True"
        if style               : params["TB style"                 ] = style
        if tb_base_params      : params["TB base parameters"       ] = tb_base_params
        if tb_refiner_params   : params["TB refiner parameters"    ] = tb_refiner_params

        if refiner_checkpoint and refiner_sampler:
            params["Denoising strength"] = refiner_cfg
            params["Hires checkpoint"  ] = refiner_checkpoint
            params["Hires upscaler"    ] = "None"
            params["Hires resize"      ] = f"{width}x{height}"
            params["Hires sampler"     ] = refiner_sampler
            params["Hires steps"       ] = max(0, refiner_steps_end-refiner_steps_start)

        return _create_infotext(positive, negative, params)


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


#======================== AUTO1111 INFOTEXT FORMAT =========================#

def _create_infotext(positive  : str | None,
                     negative  : str | None,
                     parameters: dict
                     )-> str:
    """
    Create a string compatible with the A1111's infotext to be used as metadata for the image.
    Args:
        positive  : The main prompt.
        negative  : The negative prompt.
        parameters: A dictionary containing the generation parameters in A1111 format.
    Returns:
        A string compatible with the A1111's infotext.
    """
    positive  = str(positive)                    if positive else ""
    negative  = f"\nNegative prompt: {negative}" if negative else ""
    last_line = ", ".join([f"{k}: {_infotext_quote(v)}" for k, v in parameters.items() if v is not None])
    return f"{positive}{negative}\n{last_line}".strip()


def _infotext_quote(value) -> str:
    """Quote a string for use as an infotext value."""
    str_value = str(value)
    if (',' in str_value) or ('\n' in str_value) or (':' in str_value):
        str_value = json.dumps(str_value, ensure_ascii=False)
    return str_value


def _a1111_sampler_name(comfy_sampler: str | None, comfy_scheduler: str | None, /,*, include_unsupported: bool = False) -> str:
    """Returns the sampler name in A1111 format to be used in the infotext."""
    DEFAULT_SAMPLER   = "Euler"
    DEFAULT_SCHEDULER = "Normal"

    comfy_sampler   = str(comfy_sampler).strip()   if comfy_sampler   is not None else ""
    comfy_scheduler = str(comfy_scheduler).strip() if comfy_scheduler is not None else ""
    if not comfy_sampler:
        return None
    if not comfy_scheduler:
        comfy_scheduler = "normal"

    # try to find the sampler name by the combinations
    # of sampler+scheduler from the old predefined samplers of A1111
    if comfy_sampler and comfy_scheduler:
        sampler_scheduler = f"{comfy_sampler} {comfy_scheduler}"
        a1111_sampler     = _A1111_SAMPLER_BY_COMFY_SAMPLER_SCHEDULER.get(sampler_scheduler)
        if a1111_sampler:
            return a1111_sampler

    # get compatible A1111 names from ComfyUI's sampler and scheduler
    a1111_sampler   = _A1111_SAMPLER_BY_COMFY_NAME.get(comfy_sampler, DEFAULT_SAMPLER)
    a1111_scheduler = _A1111_SCHEDULER_BY_COMFY_NAME.get(comfy_scheduler, "")

    # names that start with '!' are not supported by A1111
    if a1111_sampler.startswith("!"):
        a1111_sampler = a1111_sampler.removeprefix("!")
        if not include_unsupported:
            a1111_sampler = DEFAULT_SAMPLER
    if a1111_scheduler.startswith("!"):
        a1111_scheduler = a1111_scheduler.removeprefix("!")
        if not include_unsupported:
            a1111_scheduler = DEFAULT_SCHEDULER

    # return the combined sampler name and scheduler
    return f"{a1111_sampler} {a1111_scheduler}".strip()


def _a1111_compact_params(sampler    : str | None,
                          scheduler  : str | None,
                          steps      : int,
                          steps_start: int,
                          steps_end  : int,
                          cfg_scale  : float,
                          seed       : int
                          ) -> str:
    """Returns a line of text containing the provided parameters separated by commas."""
    if (sampler is None) or (scheduler is None) or (steps is None) or (cfg_scale is None) or (seed is None):
        return None
    try:
        compact_params = f"sampler_name:{str(sampler).strip()}, scheduler:{str(scheduler).strip()}, steps:{int(steps)}, start_at_step:{int(steps_start or 0)}, end_at_step:{int(steps_end or steps)}, cfg:{round(float(cfg_scale),2)}, noise_seed:{int(seed)}"
    except ValueError:
        return None
    return compact_params



#========================= UNIFIED PROMPT EDITION ==========================#
import copy

def _update_unified_prompt(workflow: dict, random_seed: int) -> dict:
    """
    Updates the unified prompt within a workflow by replacing random seed values.

    This function searches for and replaces occurrences of "--seed random"
    with the provided random seed value. It handles different types of workflows,
    such as `GenParams` and `ComfyUI workflows`, etc.

    Args:
        workflow   (dict): The workflow to update.
        random_seed (int): The integer value to replace the "random" seed with.

    Returns:
        The updated workflow.
    """
    NODE_TYPE = "UnifiedPromptInput //TinyBreaker"
    workflow = copy.deepcopy(workflow)

    # genparams
    if "user.prompt" in workflow:
        workflow["user.prompt"] = _replace_random_values(workflow["user.prompt"],
                                                         random_seed=random_seed)
    # comfyui-workflow
    elif "nodes" in workflow:
        nodes = workflow["nodes"]
        if isinstance(nodes,list):
            for node in nodes:
                if isinstance(node,dict) and node.get('type') == NODE_TYPE:
                    widgets_values = node.get('widgets_values')
                    if isinstance(widgets_values, list):
                        for i in range(len(widgets_values)):
                            widgets_values[i] = _replace_random_values(widgets_values[i],
                                                                       random_seed=random_seed)
    # comfyui-prompt
    else:
        for id, node in workflow.items():
            if isinstance(node,dict) and node.get("class_type") == NODE_TYPE:
                inputs = node.get("inputs")
                if isinstance(inputs,dict) and "text" in inputs:
                    inputs["text"] = _replace_random_values(inputs["text"],
                                                            random_seed=random_seed)
    return workflow


def _replace_random_values(text: str, /,*, random_seed: int | None) -> str:
    """
    Replaces "random" seed values with the provided seed.
    Returns the updated text with the seed values replaced.
    """
    if not isinstance(text, str) or not text:
        return text
    if random_seed is None:
        return text

    # analyze one by one the arguments in the prompt
    # replacing the "--seed random" with the actual seed value
    prompt, args = split_prompt_and_args(text)
    for i in range(len(args)):
        arg = args[i]
        if arg.startswith("seed "):
            args[i] = arg.replace("random", str(random_seed))

    # rejoin the prompt and arguments
    text = join_prompt_and_args(prompt, args)
    return text

