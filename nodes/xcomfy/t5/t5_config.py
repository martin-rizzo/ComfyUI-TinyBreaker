"""
File    : xcomfy/pixart_t5.py
Purpose : Custom versions of ComfyUI native configuration classes (for T5 tokenizer/encoder).
          (include some small tweaks to improve a few details)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 2, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import comfy.text_encoders.t5
from comfy        import sd1_clip
from transformers import T5TokenizerFast


#======================== SD3/PIXAR T5 MODEL CONFIG ========================#

class T5XXLModel_SD3(sd1_clip.SDClipModel):
    """
    A custom version of the T5XXLModel class that allows injecting any T5
    encoder model via the key "t5_encoder_class" in the `model_options` dictionary.
    The original ComfyUI implementation can be found here:
    - https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/text_encoders/sd3_clip.py#L10
    """
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=False, model_options={}):
        t5_config_dir         = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config")
        textmodel_json_config = os.path.join(t5_config_dir, "t5_config_xxl.json")
        t5_encoder_class      = comfy.text_encoders.t5.T5

        print("##>> _T5XXLModel dtype:", dtype)

        if "t5xxl_scaled_fp8" in model_options:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = model_options["t5xxl_scaled_fp8"]

        if "t5_encoder_class" in model_options:
            t5_encoder_class = model_options.pop("t5_encoder_class")

        super().__init__(device                 = device,
                         layer                  = layer,
                         layer_idx              = layer_idx,
                         textmodel_json_config  = textmodel_json_config,
                         dtype                  = dtype,
                         special_tokens         = {"end": 1, "pad": 0},
                         model_class            = t5_encoder_class,
                         enable_attention_masks = attention_mask,
                         return_attention_masks = attention_mask,
                         model_options          = model_options)


class T5XXLModel_PA(T5XXLModel_SD3):
    """
    A variant of T5XXLModel_SD3 but specialized for the PixArt model.
    In the same way as T5XXLModel this class allows injecting any T5 model
    via the key "t5_encoder_class" in the `model_options` dictionary.
    The original ComfyUI implementation can be found here:
    - https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/text_encoders/pixart_t5.py#L10
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_empty_tokens(self, special_tokens, *args, **kwargs):
        special_tokens = special_tokens.copy()
        special_tokens.pop("end")
        return comfy.sd1_clip.gen_empty_tokens(special_tokens, *args, **kwargs)


#======================== PIXAR T5 TOKENIZER CONFIG ========================#

class T5XXLTokenizer_PA(sd1_clip.SDTokenizer):
    """
    The PixArt T5 Tokenizer configuration class

    Nothing was modified, the configuration is the same as native ComfyUI, but
    the class stays here just in case we want to customize any configuration
    in the future

    The original ComfyUI implementation can be found here:
    - https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/text_encoders/pixart_t5.py#L24
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config")
        super().__init__(tokenizer_path,
                         embedding_directory = embedding_directory,
                         pad_with_end        = False,
                         embedding_size      = 4096,
                         embedding_key       = "t5xxl",
                         tokenizer_class     = T5TokenizerFast,
                         has_start_token     = False,
                         pad_to_max_length   = False,
                         max_length          = 99999999,
                         min_length          = 1,
                         )

class PixArtTokenizer(sd1_clip.SD1Tokenizer):
    """ComfyUI wrapper for T5XXLTokenizer_PA (?)"""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory = embedding_directory,
                         tokenizer_data      = tokenizer_data,
                         clip_name           = "t5xxl",
                         tokenizer           = T5XXLTokenizer_PA,
                         )

