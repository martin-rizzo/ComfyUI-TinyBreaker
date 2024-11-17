"""
File    : vae_loader.py
Purpose : Node to load any VAE model including `Tiny AutoEncoder` (TAESD) variants.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 16, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import torch
from ..utils.directories import VAE_DIR
from comfy.utils import load_torch_file
from comfy.sd    import VAE


class VAELoader:
    TITLE       = "xPixArt | VAE Loader"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Load VAE models including `Tiny AutoEncoder` (TAESD) variants."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (VAE_DIR.get_filename_list(), ),
                }
            }

    #-- FUNCTION --------------------------------#
    FUNCTION = "load_vae"
    RETURN_TYPES = ("VAE",)

    def load_vae(self, vae_name):

        # load VAE model state dictionary from file
        vae_path   = VAE_DIR.get_full_path_or_raise(vae_name)
        state_dict = load_torch_file(vae_path)

        # fix `Tiny AutoEncoder` state dictionary if necessary
        if self.is_taesd(state_dict):
            state_dict = self.fix_taesd_state_dict(state_dict, filename=os.path.basename(vae_path))

        vae = VAE(sd=state_dict)
        return (vae,)



    @staticmethod
    def is_taesd(state_dict: dict) -> bool:
        """
        Determine if the given state dictionary corresponds to a Tiny AutoEncoder (TAESD) model.
        """

        # recognize:
        #   taesd_decoder.safetensors, taesd_encoder.safetensors,
        #   taesdxl_decoder.safetensors, taesdxl_encoder.safetensors
        if "3.conv.4.bias"   in state_dict and \
           "8.conv.0.weight" in state_dict:
           return True

        # recognize:
        #   diffusion_pytorch_model.safetensors (SD, SDXL, SD3, FLUX.1)
        if "decoder.layers.3.conv.4.bias"   in state_dict and \
           "decoder.layers.8.conv.0.weight" in state_dict:
            return True
        if "encoder.layers.4.conv.4.bias"   in state_dict and \
           "encoder.layers.8.conv.0.weight" in state_dict:
            return True

        # recognize any model whose root is "taesd_"
        for key in state_dict.keys():
            if key.startswith("taesd_"):
                return True

        # none of the above conditions are met
        # therefore, it does not appear to be a `Tiny AutoEncoder` model
        return False


    @staticmethod
    def fix_taesd_state_dict(tae_tensors: dict, filename: str) -> dict:
        """
        Adjust the state dictionary to match the expected format for a Tiny AutoEncoder (TAESD) model.
        """
        encoder_layer_offset = 0
        decoder_layer_offset = 0

        # check if it's necessary to adjust 'layer_number'
        if "decoder.layers.0.weight" in tae_tensors:
            decoder_layer_offset = 1
       
        state_dict = {}
        for key, tensor in tae_tensors.items():
            prefix, layer_number = "", ""

            # convert "encoder." -> "taesd_encoder."
            # and     "decoder." -> "taesd_decoder."
            if key.startswith("encoder."):
                key    = key[len("encoder."):]
                prefix = "taesd_encoder."
            elif key.startswith("decoder."):
                key    = key[len("decoder."):]
                prefix = "taesd_decoder."

            # extract 'root' which is the first section of the tensor name
            root = f"{prefix}{layer_number}{key}".split('.',1)[0]

            # accept only tensors with root = "taesd_*"
            if not root.startswith("taesd_"):
                continue

            # remove any prefix "layers."
            if key.startswith("layers."):
                key = key[len("layers."):]

            # extract the layer_number if the prefix is a number followed by '.'
            number, _subkey = key.split('.',1)
            if number.isdigit():
                key = _subkey
                # adjust `layer_number` according to `*_layer_offset`
                if root == "taesd_encoder":
                    layer_number = f"{int(number) + encoder_layer_offset}."
                elif root == "taesd_decoder":
                    layer_number = f"{int(number) + decoder_layer_offset}."

            # assign the tensor to its new name
            state_dict[f"{prefix}{layer_number}{key}"] = tensor

        # add vae_scale and vae_shift if they are missing
        if "vae_scale" not in state_dict and "vae_shift" not in state_dict:
            lower_case_filename = filename.lower()
            if "sdxl" in lower_case_filename:
                state_dict["vae_scale"] = torch.tensor(0.13025)
                state_dict["vae_shift"] = torch.tensor(0.0)
            elif "sd3" in lower_case_filename:
                state_dict["vae_scale"] = torch.tensor(1.5305)
                state_dict["vae_shift"] = torch.tensor(0.0609)
            elif "f1" in lower_case_filename or "flux" in lower_case_filename:
                state_dict["vae_scale"] = torch.tensor(0.3611)
                state_dict["vae_shift"] = torch.tensor(0.1159)
            else:
                state_dict["vae_scale"] = torch.tensor(0.18215)
                state_dict["vae_shift"] = torch.tensor(0.0)

        return state_dict

