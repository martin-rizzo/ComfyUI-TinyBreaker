"""
File    : save_prompt_embedding.py
Purpose : 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 12, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
from safetensors.torch import save_file as save_safetensors
from ...utils.directories     import PROMPT_EMBEDS_DIR
MAX_RESOLUTION=16384


class SavePromptEmbedding:
    TITLE       = "xPixArt | Save Prompt Embedding"
    CATEGORY    = "xPixArt/testing"
    DESCRIPTION = ""
    OUTPUT_NODE = True

    def __init__(self):
        self.default_ext = ".safetensors"

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in_positive": ("CONDITIONING", ),
                "in_negative": ("CONDITIONING", ),
                "in_width"   : (   "INT", {"default": 1024, "min": 1  , "max": MAX_RESOLUTION, "step": 1}),
                "in_height"  : (   "INT", {"default": 1024, "min": 1  , "max": MAX_RESOLUTION, "step": 1}),
                "in_seed"    : (   "INT", {"default":    0, "min": 0  , "max": 0xffffffffffffffff}),
                "in_steps"   : (   "INT", {"default":   16, "min": 1  , "max": 10000         , "step": 1}),
                "in_cfg"     : ( "FLOAT", {"default":  4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "filename": ("STRING", {"default": "prompt"})
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION     = "save"
    RETURN_TYPES = ()
    def save(self, in_positive, in_negative, in_width, in_height, in_seed, in_steps, in_cfg, filename):
        # in_positive = [ [cond,{extra_conds}]], ... ]
        # in_negative = [ [cond,{extra_conds}]], ... ]

        tensors  = { }
        metadata = { }

        # "prompt.positive" and "prompt.positive{1..1000}"
        #   - prompt.positive
        #   - prompt.positive_attn_mask
        multiple = len(in_positive)>1
        for i, cond_unit in enumerate(in_positive):
            prompt_key = f"prompt.positive{i}" if multiple else "prompt.positive"
            tensors[prompt_key] = cond_unit[0] # 0: cond
            extra_conds         = cond_unit[1] # 1: extra_conds
            if "cond_attn_mask" in extra_conds:
                attn_mask_key = f"{prompt_key}_attn_mask"
                tensors[attn_mask_key] = extra_conds["cond_attn_mask"]

        # "prompt.negative" and "prompt.negative{1..1000}"
        #   - prompt.negative
        #   - prompt.negative_attn_mask
        multiple = len(in_negative)>1
        for i, cond_unit in enumerate(in_negative):
            prompt_key = f"prompt.negative{i}" if multiple else "prompt.negative"
            tensors[prompt_key] = cond_unit[0] # 0: cond
            extra_conds         = cond_unit[1] # 1: extra_conds
            if "cond_attn_mask" in extra_conds:
                attn_mask_key = f"{prompt_key}_attn_mask"
                tensors[attn_mask_key] = extra_conds["cond_attn_mask"]

        # METADATA "parameters.*"
        metadata["parameters.steps"    ] = str(in_steps)
        metadata["parameters.width"    ] = str(in_width)
        metadata["parameters.height"   ] = str(in_height)
        metadata["parameters.cfg"      ] = str(in_cfg)
        metadata["parameters.seed"     ] = str(in_seed)
        # METADATA "pt5tokenizer.*"
        metadata["pt5tokenizer.mode"   ] = "bug_120chars"
        metadata["pt5tokenizer.padding"] = "false"

        # armar el path completo
        _, extension = os.path.splitext(filename)
        if not extension:
            filename += self.default_ext
        filepath = PROMPT_EMBEDS_DIR.get_full_path(filename, for_save=True)

        # almacenar como safetensor
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_safetensors(tensors, filepath, metadata=metadata)
        return { }

