"""
File    : placeholder_replacement.py
Desc    : Node that replaces placeholders in a text with corresponding values.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 13, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""


class PlaceholderReplacer:
    TITLE       = "xPixArt | Placeholder Replacer"
    CATEGORY    = "xPixArt/strings"
    DESCRIPTION = "Replace placeholders in strings with their corresponding values."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template"         : ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The template text with placeholders to be replaced."}),
                "placeholder_name" : ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The name of the placeholder.", "default": "$var"}), 
                "var1"             : ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "var2"             : ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "var3"             : ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }
    
    #-- FUNCTION --------------------------------#
    FUNCTION = "replace_placeholders"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT",)

    def replace_placeholders(self, template, placeholder_name, var1, var2, var3):
        placeholders = {
            f"{placeholder_name}1" : var1,
            f"{placeholder_name}2" : var2,
            f"{placeholder_name}3" : var3,
        }
        # replace each placeholder with its corresponding value
        for placeholder, value in placeholders.items():
            template = template.replace(placeholder, value)  
        return (template,)

