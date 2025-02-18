"""
File    : safetensors.py
Purpose : Safetensors file handling functions.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import json
import struct
from typing import Union


def normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix by ensuring it ends with a dot when it's not empty."""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


def filter_state_dict(state_dict: dict, prefix: str) -> dict:
    """
    Extracts a specific section from a state dictionary based on a given prefix.
    Args:
        state_dict (dict): A dictionary containing tensors representing the model's state.
        prefix      (str): The prefix string used to identify the desired section within the state dictionary.
    Returns:
        A new dictionary containing only the tensors with names starting with the specified prefix.
        The keys in the returned dictionary are the original keys with the prefix removed.
    """
    prefix = normalize_prefix(prefix)
    return { name.removeprefix(prefix): tensor for name, tensor in state_dict.items() if name.startswith(prefix) }


def load_safetensors_header(file_path: str, size_limit: int = 67108864) -> dict:
    """
    Load the header of a SafeTensors file.
    Args:
        file_path  (str): Path to the .safetensors file.
        size_limit (int): Maximum allowed size for the header.
    Returns:
        Dictionary containing the header data.
    """
    try:
        with open(file_path, "rb") as f:
            header = None

            # read the length of the header (8 bytes)
            # and decode the header data
            header_length = struct.unpack("<Q", f.read(8))[0]
            if header_length <= size_limit:
                header_data = f.read(header_length)
                header      = json.loads(header_data)

            return header

    except (ValueError, json.JSONDecodeError):
        filename = os.path.basename(file_path)
        raise ValueError(f"The file `{filename}` does not have a valid safetensors header.")
    except IOError:
        filename = os.path.basename(file_path)
        raise IOError(f"Error opening or reading the file `{filename}`.")


def estimate_model_params(file_path     : str,
                          tensors_prefix: str = ""
                          ) -> int:
    """
    Estimates the number of parameters of a model from a SafeTensors file.
    La funcion no carga ningun tensor, realiza el calculo con la informacion del header.
    Args:
        file_path      (str): Path to the .safetensors file containing the model weights.
        tensors_prefix (str): Prefix to filter tensors that contribute to the parameter count;
                              defaults to an empty string.
    Returns:
        int: Estimated number of parameters of the model.
    """

    # the prefix should end with a '.' to match the full tensor name correctly
    if tensors_prefix and not tensors_prefix.endswith('.'):
        tensors_prefix += '.'

    number_of_params = 0
    header = load_safetensors_header(file_path)
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


def detect_format_and_prefix(safetensors: Union[str, dict],
                             format_a   : list[str, list] = None,
                             format_b   : list[str, list] = None,
                             format_c   : list[str, list] = None
                             ) -> tuple[str, str]:
    """
    Detects the format and prefix of a .safetensors file based on its header.

    Args:
        safetensors (Union[str, dict]): Path to the .safetensors file or its tensor dictionary.
        format_a  (list[str, list]): Definition containing the format name and a list of tensor names.
        format_b  (list[str, list]): Definition containing the format name and a list of tensor names.
        format_c  (list[str, list]): Definition containing the format name and a list of tensor names.

    Returns:
        tuple[str, str]: A tuple containing the detected format name and the prefix.

    Example:
        format_a = ("Format A", ["model.", "encoder."]        )
        format_b = ("Format B", ["decoder.", "generator."]    )
        format_c = ("Format C", ["discriminator.", "critic."] )
        format_name, prefix = detect_format_and_prefix("path/to/model.safetensors", format_a, format_b, format_c)
        print(f"Format: {format_name}, Prefix: {prefix}")
    """
    if isinstance(safetensors, str):
        safetensors = load_safetensors_header(safetensors)
    if not isinstance(safetensors, dict):
        return '', ''

    # get the list of required tensors for each format
    all_tensors_a = format_a[1] if format_a else None
    all_tensors_b = format_b[1] if format_b else None
    all_tensors_c = format_c[1] if format_c else None

    # get the first required tensor of each format
    req_tensor_a = all_tensors_a[0] if all_tensors_a else None
    req_tensor_b = all_tensors_b[0] if all_tensors_b else None
    req_tensor_c = all_tensors_c[0] if all_tensors_c else None

    # filter out None values from the list of required tensors
    req_tensors = tuple(filter(None, [req_tensor_a, req_tensor_b, req_tensor_c]))

    # check if any of the required tensors are present in the safetensors keys
    safetensors_keys = safetensors.keys()
    for key in safetensors_keys:
        if not key.endswith(req_tensors):
            continue

        if req_tensor_a and key.endswith(req_tensor_a):
            prefix = key[:-len(req_tensor_a)]
            if all( prefix+tensor_name in safetensors_keys for tensor_name in all_tensors_a ):
                # all required tensors from format_a are present!
                # return the format name and prefix
                return format_a[0], prefix

        if req_tensor_b and key.endswith(req_tensor_b):
            prefix = key[:-len(req_tensor_b)]
            if all( prefix+tensor_name in safetensors_keys for tensor_name in all_tensors_b ):
                # all required tensors from format_b are present!
                # return the format name and prefix
                return format_b[0], prefix

        if req_tensor_c and key.endswith(req_tensor_c):
            prefix = key[:-len(req_tensor_c)]
            if all( prefix+tensor_name in safetensors_keys for tensor_name in all_tensors_c ):
                # all required tensors from format_c are present!
                # return the format name and prefix
                return format_c[0], prefix

    # requeried tensors not found, return empty strings
    return '', ''
