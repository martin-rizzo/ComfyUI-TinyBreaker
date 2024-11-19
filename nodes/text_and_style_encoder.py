"""
File    : text_and_style_encoder.py
Purpose : Node to encode prompts applying preconfigured styles.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 18, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from ..utils.directories     import STYLES_DIR
from ..core.style_collection import StyleCollection


class TextAndStyleEncoder:
    TITLE       = "xPixArt | Text and Style Encoder"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Generate text embeddings modified by the selected image style."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_name"     : (cls.get_style_names(), {"tooltip": "The name of the style to apply."}),
               #"style_strength" : ("FLOAT", { "default": 1.0, "min": 0.0, "max": 2.0 }),
                "prompt"         : ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt to be encoded."}), 
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt to be encoded."}), 
                "clip"           : ("CLIP"  , {"tooltip": "The T5 model used for encoding the text."}),
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING"     )
    RETURN_NAMES = ("positive"    , "negative"    , "prompt_text")

    @classmethod
    def encode(cls, style_name, prompt, negative_prompt, clip):
        style_strength = 1.0
        refiner_text   = prompt
        negative       = negative_prompt

        style = cls.get_style(style_name)
        if not style:
            raise ValueError(f"Unknown style type '{style_name}'")

        prompt_cond   = cls.make_conditioning(prompt  , clip, style, style_strength, type="prompt"  )
        negative_cond = cls.make_conditioning(negative, clip, style, style_strength, type="negative")
        refiner_text  = style.apply_to_text(refiner_text, type="refiner")
        print(f"##>> refiner: {refiner_text}")
        print("-----------")

        return ([prompt_cond], [negative_cond], refiner_text)



    #-[ internal functions ]---------------------#

    styles = {"None": None}

    @classmethod
    def get_style(cls, style_name):
        return cls.styles.get(style_name, None) 


    @classmethod
    def get_style_names(cls):
        cls.styles  = StyleCollection.from_directory( STYLES_DIR.paths[0] )
        style_names = list( cls.styles.keys() )
        return style_names


    @classmethod
    def make_conditioning(cls, text, clip, style, style_strength, type="prompt"):
        """
        Encode the given text using the provided CLIP model.
        (if a style is specified, apply the style to text and resulting embedding)
        
        Args:
            text  (str): The input text to encode.
            clip (CLIP): The CLIP model used for encoding the text.
            style (Style): The style to apply to the text before encoding.
            style_strength (float): The strength of the style effect.
            type (str): The type of conditioning ("prompt", "negative" or "refiner").
        
        Returns:
            list: A list containing the encoded condition and any additional output from the CLIP model.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # apply style to text before encoding
        if style and style_strength > 0.0:
            text = style.apply_to_text(text, 
                                       type=type)
            print(f"##>> {type}: {text}")
            print("-----------")


        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond   = output.pop("cond")

        # apply style to embedding after encoding
        if style and style_strength > 0.0:
            cond  = style.apply_to_tensor(cond, style_strength, type=type)

        return [cond, output]
