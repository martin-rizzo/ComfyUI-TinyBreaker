"""
File    : save_image.py
Purpose : Node for saving a generated images to disk with A1111/CivitAI embedded metadata.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import time
import numpy as np
import folder_paths
from PIL                import Image
from .core.gen_params   import GenParams
from .common            import create_a1111_params, \
                               create_png_info,     \
                               expand_variables


class SaveImage:
    TITLE       = "💪TB | Save Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Saves images embedding A1111/CivitAI metadata within PNG files. This facilitates easy extraction of prompts and settings through widely available tools."
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
        a1111_params = create_a1111_params(genparams, image_width, image_height)
        pnginfo      = create_png_info(prompt        = prompt,
                                       extra_pnginfo = extra_pnginfo,
                                       a1111_params  = a1111_params)

        # resolve filename_prefix and get full path to save images
        filename_prefix = expand_variables(f"{filename_prefix}{self.extra_prefix}",
                                           time       = time.localtime(),
                                           extra_vars = genparams
                                           )
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


