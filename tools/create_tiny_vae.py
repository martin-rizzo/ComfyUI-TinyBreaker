"""
File    : create_tiny_vae.py
Purpose : Command-line tool to create a tiny Variational Autoencoder (VAE) model.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 23, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import json
import struct
import argparse
from safetensors       import safe_open
from safetensors.numpy import save_file

#---------------------------- COLORED MESSAGES -----------------------------#

GRAY   = "\033[90m"
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
CYAN   = '\033[96m'
DEFAULT_COLOR = '\033[0m'
FATAL_ERROR_CODE = 1

def disable_colors():
    """Disables colored messages."""
    global GRAY, RED, GREEN, YELLOW, BLUE, CYAN, DEFAULT_COLOR
    GRAY, RED, GREEN, YELLOW, BLUE, CYAN, DEFAULT_COLOR = '', '', '', '', '', '', ''

def warning(message: str, *info_messages: str) -> None:
    """Displays and logs a warning message to the standard error stream."""
    print(f"{CYAN}[{YELLOW}WARNING{CYAN}]{YELLOW} {message}{DEFAULT_COLOR}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {YELLOW}{info_message}{DEFAULT_COLOR}", file=sys.stderr)
    print()

def error(message: str, *info_messages: str) -> None:
    """Displays and logs an error message to the standard error stream."""
    print(f"{CYAN}[{RED}ERROR{CYAN}]{RED} {message}{DEFAULT_COLOR}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {RED}{info_message}{DEFAULT_COLOR}", file=sys.stderr)
    print()

def fatal_error(message: str, *info_messages: str) -> None:
    """Displays and logs an fatal error to the standard error stream and exits.
    Args:
        message       : The fatal error message to display.
        *info_messages: Optional informational messages to display after the error.
    """
    error(message)
    for info_message in info_messages:
        print(f" {CYAN}\u24d8  {info_message}{DEFAULT_COLOR}", file=sys.stderr)
    exit(FATAL_ERROR_CODE)


#--------------------------------- HELPERS ---------------------------------#

def get_safetensors_header(file_path : str,
                           size_limit: int = 67108864
                           ) -> dict:
    """
    Returns a dictionary with the safetensors file header for fast content validation.
    Args:
        file_path  (str): Path to the .safetensors file.
        size_limit (int): Maximum allowed size for the header (a protection against large headers)
    """
    try:
        # verify that the file has at least 8 bytes (the minimum size for a header)
        if os.path.getsize(file_path) < 8:
            return []
        
        # read the first 8 bytes to get the header length and decode the header data
        with open(file_path, "rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            if header_length > size_limit:
                return []
            header = json.loads( f.read(header_length) )
            return header
        
    # handle exceptions that may occur during header reading or decoding
    except (ValueError, json.JSONDecodeError, IOError):
        return []


def get_tensor_prefix(state_dict: dict, postfix: str, not_contain: str = None) -> str:
    """
    Returns the prefix of a key in the state dictionary that matches the given postfix.
    Args:
        state_dict (dict): The model parameters as a dictionary.
        postfix  (str): The suffix to match at the end of the key.
    """
    # iterate over all keys in the state dictionary
    for key in state_dict.keys():
        # check if the key ends with the given postfix
        if key.endswith(postfix):
            if (not_contain is not None) and (not_contain in key):
                continue
            return key[:-len(postfix)]
        
    # if no key matches the postfix, return an empty string
    return ""


def load_tensors(path         : str,
                 prefix       : str,
                 target_prefix: str = ""
                 ):
    """
    Load tensors with the specified prefix from a safetensors file.
    
    Args:
        path          (str) : The path to the safetensors file.
        prefix        (str) : The prefix of the tensors to load.
        target_prefix (str) : The prefix used as replacement of the original prefix.
                              If empty, the original prefix is removed and not replaced.
    Returns:
        dict: A dictionary containing the loaded tensors.
    """
    # ensure the prefixes end with a dot
    if prefix and not prefix.endswith('.'):
        prefix += '.'
    if target_prefix and not target_prefix.endswith('.'):
        target_prefix += '.'

    # load the tensors from the file with the specified prefix
    tensors = {}
    prefix_len = len(prefix)
    with safe_open(path, framework="numpy", device='cpu') as f:
        for key in f.keys():
            if key.startswith(prefix):
                target_key = target_prefix + key[prefix_len:]
                tensors[target_key] = f.get_tensor(key)

    return tensors


def find_unique_path(path: str) -> str:
    """
    Returns the first available path to not overwrite an existing file.
    Args:
        file_path (str): The initial file path.
    """
    if not os.path.exists(path):
        return path
    base_name, extension = os.path.splitext(path)
    number = 1
    while True:
        new_path = f"{base_name}_{number:02d}{extension}"
        if not os.path.exists(new_path):
            return new_path
        number += 1


#----------------------------- IDENTIFICATION ------------------------------#

def is_taesd(state_dict: dict) -> bool:
    """
    Returns True if the model parameters correspond to a Tiny AutoEncoder (TAESD) model.
    Args:
        state_dict (dict): The model parameters as a dictionary.
    """
    # recognize the following files based on their structure:
    #   - taesd_decoder.safetensors
    #   - taesd_encoder.safetensors
    #   - taesdxl_decoder.safetensors
    #   - taesdxl_encoder.safetensors
    #
    if  "3.conv.4.bias"   in state_dict and \
        "8.conv.0.weight" in state_dict:
        return True

    # recognize the following diffusers files based on their structure:
    #   - diffusion_pytorch_model.safetensors (SD, SDXL, SD3 and FLUX.1 version)
    #
    if  "decoder.layers.3.conv.4.bias"   in state_dict and \
        "decoder.layers.8.conv.0.weight" in state_dict:
        return True
    if  "encoder.layers.4.conv.4.bias"   in state_dict and \
        "encoder.layers.8.conv.0.weight" in state_dict:
        return True

    # recognize any model whose tensor root name starts with some TAESD-related names
    for key in state_dict.keys():
        if key.startswith( ("taesd", "taesdxl", "taesd3", "taef1")  ):
            return True

    # none of the above conditions are met
    # therefore, it does not appear to be a `Tiny AutoEncoder` model
    return False


def is_taesd_with_role(file_path: str, state_dict: dict, role: str) -> bool:
    """
    Returns True if the model parameters correspond to a Tiny AutoEncoder (TAESD) model with a specific role.
    Args:
        file_path  (str) : The path to the model file.
        state_dict (dict): The model parameters or safetensors header.
        role       (str) : The role of the model, either 'encoder' or 'decoder'.
    """
    assert role in ("encoder", "decoder"), "Invalid role. Must be 'encoder' or 'decoder'."

    # names of tensors that betray the role of the model (encoder/decoder)
    ENCODER_TENSOR_SUBNAMES = {
        "encoder" : ("encoder", ),
        "decoder" : ("decoder", )
    }

    # check if state_dict contains any keys related to the specified role
    if not state_dict or not is_taesd(state_dict):
        return False
    subnames = ENCODER_TENSOR_SUBNAMES[role]
    for key in state_dict.keys():
        if any(subname in key for subname in subnames):
            return True
        
    # how last, check if the filename itself contains the role information
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    return role in file_name.lower()


def find_taesd_with_role(input_files: list[str], role: str) -> tuple[str, str]:
    """
    Find the Tiny AutoEncoder (TAESD) model with a specific role from a list of input files.
    
    Args:
        input_files (list[str]): List of input file paths.
        role            (str)  : The role of the model, either 'encoder' or 'decoder'.
    Returns:
        tuple[str, str]: A tuple containing the taesd model filename and its tensor prefix.
    """
    assert role in ("encoder", "decoder"), "Invalid role. Must be 'encoder' or 'decoder'."
    oposite_role = "decoder" if role == "encoder" else "encoder"
    
    file_path     = ""
    tensor_prefix = ""
    for file in input_files:
        header = get_safetensors_header(file)
        if is_taesd_with_role(file, header, role):
            file_path     = file
            tensor_prefix = get_tensor_prefix(header, ".3.conv.4.bias", not_contain=oposite_role)
            break
    
    return (file_path, tensor_prefix)


#-------------------------------- BUILDING ---------------------------------#

def build_tiny_vae(encoder_path_and_prefix: tuple[str, str],
                   decoder_path_and_prefix: tuple[str, str],
                   ) -> dict:
    """
    Build a Tiny VAE model using the provided encoder and decoder paths.
    
    Args:
        encoder_path_and_prefix (tuple[str, str]): The path to the encoder file and its tensor prefix.
        decoder_path_and_prefix (tuple[str, str]): The path to the decoder file and its tensor prefix.
        
    Returns:
        dict: The Tiny VAE model parameters.
    """

    encoder_tensors = load_tensors(path   = encoder_path_and_prefix[0],
                                   prefix = encoder_path_and_prefix[1],
                                   target_prefix = "taesd_encoder")
    
    decoder_tensors = load_tensors(path   = decoder_path_and_prefix[0],
                                   prefix = decoder_path_and_prefix[1],
                                   target_prefix = "taesd_decoder")
    
    print("##>> encoder keys:", len(encoder_tensors))
    print("##>> decoder keys:", len(decoder_tensors))
    
    # combine the encoder and decoder parameters into a single dictionary
    tiny_vae_params = {**encoder_tensors, **decoder_tensors}
    return tiny_vae_params


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args: list=None, parent_script: str=None):

    # allow this command to be a subcommand of a larger tool (future expansion?)
    prog = None
    if parent_script:
        prog = parent_script + ' ' + os.path.basename(__file__).split('.')[0]

    # start parsing the arguments
    parser = argparse.ArgumentParser(prog=prog,
        description="Create a tiny VAE from input files.",
        formatter_class=argparse.RawTextHelpFormatter,
        )
    parser.add_argument("input_files", nargs="+", help="Input files to process")
    parser.add_argument("-o", "--output_dir", help="Output directory for the VAE")

    args = parser.parse_args()

    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)

    # find the encoder and decoder files and their tensor prefixes
    encoder_path, encoder_source_prefix = find_taesd_with_role(args.input_files, role="encoder")
    decoder_path, decoder_source_prefix = find_taesd_with_role(args.input_files, role="decoder")
    if not encoder_path:
        fatal_error("No TAESD encoder model found.")
    if not decoder_path:
        fatal_error("No TAESD decoder model found.")

    print(f"Encoder file: {encoder_path}, Tensor prefix: {encoder_source_prefix}")
    print(f"Decoder file: {decoder_path}, Tensor prefix: {decoder_source_prefix}")

    state_dict = build_tiny_vae(encoder_path_and_prefix = (encoder_path, encoder_source_prefix),
                                decoder_path_and_prefix = (decoder_path, decoder_source_prefix),
                                )

    # find a unique path for the output file
    output_file_path = "taesd_vae.safetensors"
    if args.output_dir:
        output_file_path = os.path.join(args.output_dir, output_file_path)
    output_file_path = find_unique_path(output_file_path)

    # save the state dict to a file
    print(f"Saving VAE to {output_file_path}")
    save_file(state_dict, output_file_path)


if __name__ == "__main__":
    main()
