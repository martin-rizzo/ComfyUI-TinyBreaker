"""
File    : utils.py
Brief   : Utility functions and constants for ComfUI-x-PixArt
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 15, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-x-PixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-x-PixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model

    Copyright (c) 2024 Martin Rizzo

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 File Summary
 ============

  Directory constants:
      PIXART_CHECKPOINTS_DIR: folder storing pixart checkpoints
      PIXART_LORAS_DIR      : folder storing pixart LoRAs
      T5_CHECKPOINTS_DIR    : folder storing different T5 checkpoints
      PROMPT_EMBEDS_DIR     : folder with various prompt embeddings for testing

  These constant objects can be used as follows:
      PIXART_CHECKPOINTS_DIR.folder_name
      PIXART_CHECKPOINTS_DIR.get_filename_list()
      PIXART_CHECKPOINTS_DIR.get_full_path(filename, for_save=False, overwrite=False)

  Functions for .safetensors files:
      load_safetensors_header(safetensors_path) -> Dict
      estimate_model_params(safetensors_path, prefix) -> int

"""
import os
import json
import struct
import folder_paths

#=============================== DIRECTORIES ===============================#

class CustomDirectory:

    def __init__(self,
                 folder_name         : str,
                 supported_extensions: list = ['.safetensors'],
                 use_output_dir      : bool = False
                 ):
        self._initialized          = False
        self._folder_name          = folder_name
        self._supported_extensions = supported_extensions
        self._use_output_dir       = use_output_dir


    @property
    def folder_name(self):
        if self._initialized:
            return self._folder_name
        if self._use_output_dir:
            container_path = folder_paths.get_output_directory()
        else:
            container_path = folder_paths.models_dir
        full_path = os.path.join(container_path, self._folder_name)
        folder_paths.folder_names_and_paths[self._folder_name] = (
            [full_path],
            set(self._supported_extensions)
            )
        os.makedirs(full_path, exist_ok=True)
        self._initialized = True
        return self._folder_name


    def get_filename_list(self):
        return folder_paths.get_filename_list(self.folder_name)


    def get_full_path(self, filename, for_save=False, overwrite=False):
        # if it's a read file for loading (e.g., .safetensors)
        # then use the normal ComfyUI method
        if not for_save:
            return folder_paths.get_full_path(self.folder_name, filename)

        # on the other hand, if the file is for saving
        # then be careful not to overwrite any pre-existing file
        counter = 1
        folder_path = folder_paths.get_folder_paths(self.folder_name)[0]
        name, ext   = os.path.splitext(filename)
        file_path   = os.path.join(folder_path, f"{name}{ext}")
        while not overwrite and os.path.exists(file_path):
            file_path = os.path.join(folder_path, f"{name}_{counter}{ext}")
            counter += 1
        return file_path


PIXART_CHECKPOINTS_DIR = CustomDirectory('pixart')
PIXART_LORAS_DIR       = CustomDirectory('pixart_loras')
T5_CHECKPOINTS_DIR     = CustomDirectory('t5')
PROMPT_EMBEDS_DIR      = CustomDirectory('prompt_embeds', use_output_dir=True)


#=============================== SAFETENSORS ===============================#

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
            header_length = struct.unpack('<Q', f.read(8))[0]
            if header_length <= size_limit:
                header_data = f.read(header_length)
                header      = json.loads(header_data)

            return header

    except (ValueError, json.JSONDecodeError):
        filename = os.path.basename(safetensors_path)
        raise ValueError(f"The file '{filename}' does not have a valid safetensors header.")
    except IOError:
        filename = os.path.basename(safetensors_path)
        raise IOError(f"Error opening or reading the file '{filename}'.")


def estimate_model_params(safetensors_path: os.PathLike, tensors_prefix: str = '') -> int:
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
            shape = tensor_info.get('shape')
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

