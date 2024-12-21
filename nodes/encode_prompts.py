"""
File    : encode_prompts.py
Purpose : Node to encode positive, negative and refiner prompts applying preconfigured styles.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 18, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .utils.directories import STYLES_DIR
from .utils.system      import logger
#from .core.styles       import StyleCollection


class EncodePrompts:
    TITLE       = "xPixArt | Encode Prompts"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Generate text embeddings from positive, negative and refiner prompts applying preconfigured styles."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_name"     : (cls.get_style_names(), {"tooltip": "The name of the style to apply."}),
               #"style_strength" : ("FLOAT", { "default": 1.0, "min": 0.0, "max": 2.0 }),
                "prompt"         : ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt to be encoded."}), 
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt to be encoded."}), 
                "refiner_focus"  : ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The extra text to add to the refiner prompt."}),
                "t5"             : ("CLIP"  , {"tooltip": "The T5 model used for encoding the text."}),
                "refiner_clip"   : ("CLIP"  , {"tooltip": "The CLIP model used for encoding the refiner prompt."}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING"    )
    RETURN_NAMES = ("positive"    , "negative"    , "refiner_positive")

    @classmethod
    def encode(cls, style_name, prompt, negative_prompt, refiner_focus, t5, refiner_clip):
        
        style          = cls.get_style(style_name)
        style_strength = 1.0
        negative       = negative_prompt
        refiner        = cls.generate_refiner_prompt(refiner_focus, prompt)

        # if not style:
        #     raise ValueError(f"Unknown style type '{style_name}'")

        prompt_cond   = cls.make_conditioning(prompt  , t5          , style, style_strength, type="prompt"  )
        negative_cond = cls.make_conditioning(negative, t5          , style, style_strength, type="negative")
        refiner_cond  = cls.make_conditioning(refiner , refiner_clip, style, style_strength, type="refiner" )
        return ([prompt_cond], [negative_cond], [refiner_cond])



    #-[ internal functions ]---------------------#

    styles = {"None": None}

    @classmethod
    def generate_refiner_prompt(cls, refiner_focus, prompt):
        """Generate a refiner prompt based on the given focus and original prompt."""

        if not refiner_focus:
            return prompt.strip()
        
        elif "no-prompt" in refiner_focus:
            return refiner_focus.replace("no-prompt", "").strip()
        
        else:
            prompt        = prompt.strip()
            refiner_focus = refiner_focus.strip()
            return f"{refiner_focus}, {prompt}" if prompt else refiner_focus
        

    @classmethod
    def get_style(cls, style_name):
        #return cls.styles.get(style_name, None) 
        return None


    @classmethod
    def get_style_names(cls):
        #cls.styles  = StyleCollection.from_directory( STYLES_DIR.paths[0] )
        #style_names = list( cls.styles.keys() )
        #return style_names
        return ["INK", "PHOTO", "PIXEL ART"]


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
            text = style.apply_to_text(text, type=type)
            logger.debug(f"styled {type} = '{text}'")

        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond   = output.pop("cond")

        # apply style to embedding after encoding
        if style and style_strength > 0.0:
            cond  = style.apply_to_tensor(cond, style_strength, type=type)

        return [cond, output]
