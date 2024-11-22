"""
File    : t5_encoder.py
Purpose : Implement the node for encoding text prompts using the T5 model.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""

class T5TextEncoder:
    TITLE      = "xPixArt | T5 Text Encoder"
    CATEGORY   = "xPixArt"
    DECRIPTION = "Encode text prompts using the T5 model."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "t5"  : ("T5", )
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION = "encode"
    RETURN_TYPES    = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("The encoded conditioning")

    def encode(self, t5, text):
        t5_pack = t5
        padding = False

        if padding:
            tokens          = t5_pack.tokenize_with_weights(text,
                                                            padding=True,
                                                            padding_max_size=300)
            cond, attn_mask = t5_pack.encode_with_weights(tokens,
                                                        return_attn_mask=True
                                                        )
        else:
            tokens    = t5_pack.tokenize_with_weights(text, padding=False)
            cond      = t5_pack.encode_with_weights(tokens)
            attn_mask = None


        if attn_mask is not None:
            extra_conds = {"cond_attn_mask":attn_mask}
        else:
            extra_conds = { }

        print("## t5 cond.shape:", cond.shape)
        print("## t5 tokens:", tokens)
        print("## t5 attn_mask:", attn_mask)
        return ([[cond, extra_conds]], )

