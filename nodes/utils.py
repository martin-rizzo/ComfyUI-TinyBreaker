"""
File    : utils.py
Brief   : Utility functions used by ComfUI-xPixArt
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 15, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import json
import struct

#------------------------------- SAFETENSORS -------------------------------#

def load_safetensors_header(safetensors_path: str,
                            size_limit      : int = 67108864
                            ) -> dict:
    """
    Load the header of a SafeTensors file.

    Args:
        filename: Name of the SafeTensors file.

    Returns:
        Dict: Dictionary containing the header data.
    """
    try:
        with open(safetensors_path, "rb") as f:
            header = None

            # read the length of the header (8 bytes)
            # and decode the header data
            header_length = struct.unpack("<Q", f.read(8))[0]
            if header_length <= size_limit:
                header_data = f.read(header_length)
                header      = json.loads(header_data)

            return header

    except (ValueError, json.JSONDecodeError):
        filename = os.path.basename(safetensors_path)
        raise ValueError(f"The file `{filename}` does not have a valid safetensors header.")
    except IOError:
        filename = os.path.basename(safetensors_path)
        raise IOError(f"Error opening or reading the file `{filename}`.")


def estimate_model_params(safetensors_path: os.PathLike, tensors_prefix: str = "") -> int:
    """
    Estimates the number of parameters of a model from a SafeTensors file.
    La funcion no carga ningun tensor, realiza el calculo con la informacion del header.

    Args:
        safetensors_path (str): Path to the SafeTensors file containing the model.

    Returns:
        int: Estimated number of parameters of the model.
    """
    number_of_params = 0
    header = load_safetensors_header(safetensors_path)
    for key, tensor_info in header.items():
        if key.startswith(tensors_prefix):
            shape = tensor_info.get("shape")
            if shape:
                tensor_params = 1
                for value in shape:
                    tensor_params *= value
                number_of_params += tensor_params
    return number_of_params

    # total_params = 0
    # for tensor in tensors.values():
    #     total_params += tensor.numel()
    #
    # return total_params

