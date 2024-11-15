"""
File    : directories.py
Purpose : Defines directory paths as constants for various model types.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
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

"""
import os
import folder_paths

#---------------------------- CUSTOM DIRECTORY -----------------------------#
class CustomDirectory:
    """
    A base class to manage directories for storing and accessing files and models.
    This class will be subclassed to create specific directories for different types of models.
    """

    def __init__(self,
                 folder_name         : str,
                 parent_dir          : str,
                 supported_extensions: list,
                 ):
        """Initializes a custom directory for reading model files.
        Args:
            folder_name (str): The name of the directory.
            parent_dir  (str): The parent directory where the custom directory will be located. 
            supported_extensions (list): A list of file extensions that are considered valid.
        """
        self._initialized          = False
        self._folder_name          = folder_name
        self._supported_extensions = supported_extensions
        self._parent_dir           = parent_dir


    @property
    def folder_name(self):
        """Returns the name of the directory."""
        if self._initialized:
            return self._folder_name
        
        parent_dir = self._parent_dir or folder_paths.models_dir
        full_path  = os.path.join(parent_dir, self._folder_name)
        folder_paths.folder_names_and_paths[self._folder_name] = (
            [full_path],
            set(self._supported_extensions)
            )
        os.makedirs(full_path, exist_ok=True)
        self._initialized = True
        return self._folder_name


    def get_filename_list(self):
        """Returns a list of filenames in the directory."""
        return folder_paths.get_filename_list(self.folder_name)


    def get_full_path(self,
                      filename : str,
                      for_save : bool = False,
                      overwrite: bool = False
                      ) -> str:
        """Returns the full path to the specified file name.
        
        If `for_save` is True and the file already exists,
        a new unique name will be generated unless `overwrite` is True.
        
        Args:
            filename (str): The name of the file.
            for_save  (bool, optional): Whether this is being used for saving a new file. Defaults to False.
            overwrite (bool, optional): Whether to overwrite an existing file if it exists. Defaults to False.
        """
        # if it"s a read file for loading (e.g., .safetensors)
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


#----------------------------- MODEL DIRECTORY -----------------------------#
class ModelDirectory(CustomDirectory):

    def __init__(self,
                folder_name         : str,
                supported_extensions: list = [".safetensors"]):
        super().__init__(folder_name, 
                         parent_dir = folder_paths.models_dir,
                         supported_extensions = supported_extensions
                         )


#---------------------------- OUTPUT DIRECTORY -----------------------------#
class OutputDirectory(CustomDirectory):

    def __init__(self,
                folder_name         : str,
                supported_extensions: list = [".safetensors"]):
        super().__init__(folder_name,
                         parent_dir=folder_paths.get_output_directory(),
                         supported_extensions=supported_extensions
                         )


PIXART_CHECKPOINTS_DIR = ModelDirectory("pixart")
PIXART_LORAS_DIR       = ModelDirectory("pixart_loras")
T5_CHECKPOINTS_DIR     = ModelDirectory("t5")
PROMPT_EMBEDS_DIR      = OutputDirectory("prompt_embeds")

