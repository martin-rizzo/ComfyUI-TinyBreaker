"""
File    : load_prompt_embedding.py
Purpose : 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 10, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
# safe_open(filename, framework, device)
# --------------------------------------
#   Opens a safetensors file lazily and returns tensors as asked
#   Args:
#     filename  (str): The filename to open.
#     framework (str): The framework you want the tensors in.
#                      Supported values: "pt", "tf", "flax", "numpy".
#     device    (str): The device on which you want the tensors (default "cpu")
from safetensors import safe_open
from ..directories import PROMPT_EMBEDS_DIR


class LoadPromptEmbedding:
    TITLE       = "xPixArt | Load Prompt Embedding"
    CATEGORY    = "xPixArt/testing"
    DESCRIPTION = ""

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_name": (PROMPT_EMBEDS_DIR.get_filename_list(), ),
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION     = "load"
    RETURN_TYPES = ("CONDITIONING","CONDITIONING", "INT"  , "INT"   , "INT" ,  "INT" , "FLOAT")
    RETURN_NAMES = ("positive"    ,"negative"    , "width", "height", "seed", "steps", "cfg"  )
    def load(self, embed_name):
        positives = []
        negatives = []
        embed_path = PROMPT_EMBEDS_DIR.get_full_path(embed_name)
        with safe_open(embed_path, framework="pt", device="cpu") as st:
            metadata = st.metadata()
            _keys    = st.keys()

            # "prompt.positive" and "prompt.positive{1..1000}"
            #   - prompt.positive
            #   - prompt.positive_attn_mask
            cond_unit = self.read_cond_unit("prompt.positive", st,_keys)
            if cond_unit:
                positives.append(cond_unit)
            for i in range(1, 1000):
                cond_unit = self.read_cond_unit(f"prompt.positive{i}", st,_keys)
                if not cond_unit:
                    break
                positives.append(cond_unit)

            # "prompt.negative" and "prompt.negative{1..1000}"
            #   - prompt.negative
            #   - prompt.negative_attn_mask
            cond_unit = self.read_cond_unit("prompt.negative", st,_keys)
            if cond_unit:
                negatives.append(cond_unit)
            for i in range(1, 1000):
                cond_unit = self.read_cond_unit(f"prompt.negative{i}", st,_keys)
                if not cond_unit:
                    break
                negatives.append(cond_unit)

            # METADATA "parameters.*"
            height = int(  metadata.get("parameters.height", "1408"))
            width  = int(  metadata.get("parameters.width" ,  "944"))
            seed   = int(  metadata.get("parameters.seed"  ,    "0"))
            steps  = int(  metadata.get("parameters.steps" ,   "16"))
            cfg    = float(metadata.get("parameters.cfg"   ,  "4.0"))

        return ( positives, negatives, width, height, seed, steps, cfg )


    def read_cond_unit(self, prompt_key:str, safe_open, safe_keys=None):
        if safe_keys is None:
            safe_keys = safe_open.keys()
        if prompt_key not in safe_keys:
            return None

        attn_mask_key = f"{prompt_key}_attn_mask"

        cond = safe_open.get_tensor(prompt_key)
        cond = cond.unsqueeze(0)
        extra_conds = {}
        if attn_mask_key in safe_keys:
            extra_conds["cond_attn_mask"] = safe_open.get_tensor(attn_mask_key)

        return [cond, extra_conds]

