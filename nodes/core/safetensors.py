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
    """
    Normalizes a string ensuring it is a valid tensor name prefix.

    A valid tensor name prefix adheres to the following rules:
     - It should be a string.
     - It can be an empty string.
     - It can be an asterisk "*" (sometimes used to represent a wildcard search).
     - If it contains any names, it must end with a period ".". 

    For example:
      normalize_prefix("tensor")    => "tensor."
      normalize_prefix("resnet50.") => "resnet50."
      normalize_prefix("")          => ""
      normalize_prefix("*")         => "*"
    """
    if not isinstance(prefix, str):
        return ""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix



def filter_state_dict(state_dict: dict, prefix: str, subprefixes: list[str] = None) -> dict:
    """
    Filters a state dictionary to include only tensors whose keys match a given prefix.

    Args:
        state_dict       (dict): The state dictionary to filter.
        prefix            (str): The prefix that keys must start with to be included.
        subprefixes (list[str]): An optional list of subprefixes.
                                 If provided, keys must start with the main prefix followed by
                                 one of these subprefixes. Defaults to None.
   Returns:
        A new state dictionary containing only the filtered tensors.
        The `prefix` is removed from the keys in the returned dictionary,
        but `subprefixes` are *not*.

    Example:
        If state_dict is `{'layer1.weight': tensor1, 'layer1.bias': tensor2, 'layer2.weight': tensor3}`
        and `prefix` is 'layer1', the returned dictionary will be
            `{'weight': tensor1, 'bias': tensor2}`.

        If `prefix` is 'layer1' and `subprefixes` is ['.bias'], the returned dictionary will be
            `{'bias': tensor2}`.

    Example:
        >>> state_dict = {'layer1.weight': tensor1, 'layer1.bias': tensor2, 'layer2.weight': tensor3}
        >>> filter_state_dict(state_dict, 'layer1')
        {'weight': tensor1, 'bias': tensor2}

        >>> state_dict = {'layer1.weight': tensor1, 'layer1.bias': tensor2, 'layer2.weight': tensor3}
        >>> filter_state_dict(state_dict, 'layer1', ['.bias'])
        {'bias': tensor2}
    """
    prefix = normalize_prefix(prefix)
    if subprefixes:
        prefixes = [prefix + normalize_prefix(subprefix) for subprefix in subprefixes]
        is_matching_prefix = lambda name: any(name.startswith(prefix) for prefix in prefixes)
    else:
        is_matching_prefix = lambda name: name.startswith(prefix)

    return {name.removeprefix(prefix): tensor for name, tensor in state_dict.items() if is_matching_prefix(name)}



def prune_state_dict(state_dict: dict, prefix: str, subprefixes: list[str] = None) -> None:
    """
    Prunes a state dictionary by removing tensors whose keys match a given prefix.

    Args:
        state_dict       (dict): The state dictionary to prune.
        prefix            (str): The prefix that keys must start with to be removed.
        subprefixes (list[str]): An optional list of subprefixes.
                                 If provided, keys must start with the main prefix followed by
                                 one of these subprefixes to be removed. Defaults to None.
    Returns:
        None. The function modifies the `state_dict` in place.

    Example:
        >>> state_dict = {'layer1.weight': tensor1, 'layer1.bias': tensor2, 'layer2.weight': tensor3}
        >>> prune_state_dict(state_dict, 'layer1')
        >>> state_dict
        {'layer2.weight': tensor3}

        >>> state_dict = {'layer1.weight': tensor1, 'layer1.bias': tensor2, 'layer2.weight': tensor3}
        >>> prune_state_dict(state_dict, 'layer1', ['.bias'])
        >>> state_dict
        {'layer1.weight': tensor1, 'layer2.weight': tensor3}
    """
    prefix = normalize_prefix(prefix)
    if subprefixes:
        prefixes = [prefix + normalize_prefix(subprefix) for subprefix in subprefixes]
        is_matching_prefix = lambda name: any(name.startswith(prefix) for prefix in prefixes)
    else:
        is_matching_prefix = lambda name: name.startswith(prefix)

    # remove all matching keys from the original dict
    keys_to_remove = [key for key in state_dict if is_matching_prefix(key)]
    for key in keys_to_remove:
        del state_dict[key]



def update_state_dict(state_dict: dict, prefix: str, new_tensors: dict) -> None:
    """
    Updates a state dictionary with new tensors, adding a given prefix to the keys.

    Args:
        state_dict  (dict): The state dictionary to update.
                            This dictionary should map string keys to tensor values.
        prefix       (str): The prefix to add to the keys of the new tensors.
        new_tensors (dict): A dictionary of new tensors to add to the state dictionary.
                            This dictionary should map string keys to tensor values.
    Returns:
        None. The function modifies the `state_dict` in place.
    """
    prefix = normalize_prefix(prefix)
    for name, tensor in new_tensors.items():
        state_dict[prefix + name] = tensor
    return state_dict



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
    This function does not load any tensors; it calculates the parameter count using information from the header.

    Args:
        file_path      (str): Path to the .safetensors file containing the model weights.
        tensors_prefix (str): Prefix to filter tensors that contribute to the parameter count;
                              defaults to an empty string.

    Returns:
        Estimated number of parameters of the model.
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
