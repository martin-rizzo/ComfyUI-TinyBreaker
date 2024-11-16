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
class Directory:
    """
    A base class to manage directories for storing and accessing files and models.
    This class will be subclassed to create specific directories for different types of models.
    """

    def __init__(self,
                 folder_name : str,
                 custom      : bool = True,
                 read_only   : bool = False,
                 parent_dir  : str  = None,
                 supported_extensions: list = None,
                 ):
        """Initializes a custom directory for reading model files.
        Args:
            folder_name    (str): The name of the directory.
            custom        (bool): If True, the directory is a custom directory of this project.            
            read_only     (bool): If True, the directory is read-only and no files can be written to it.
            parent_dir     (str): The parent directory where the directory will be located if `create_folder` is True. 
            supported_extensions (list): A list of file extensions that are considered valid.
        """
        self._initialized = False
        self._folder_name = folder_name
        self._custom      = custom
        self._read_only   = read_only
        self._parent_dir  = parent_dir
        self._supported_extensions = supported_extensions


    @property
    def folder_name(self):
        """Returns the name of the directory."""
        if self._custom and not self._initialized:
            # create and initialize the custom directory
            parent_dir           = self._parent_dir or folder_paths.models_dir
            supported_extensions = self._supported_extensions or [".safetensors"]
            full_path  = os.path.join(parent_dir, self._folder_name)
            folder_paths.folder_names_and_paths[self._folder_name] = (
                [full_path],
                set(supported_extensions)
                )
            os.makedirs(full_path, exist_ok=True)

        self._initialized = True
        return self._folder_name
        

    def get_filename_list(self):
        """Returns a list of filenames in the directory."""
        return folder_paths.get_filename_list(self.folder_name)


    def get_full_path(self, filename: str) -> str:
        """Returns the full path to a file in the directory, returning None if the file is not found."""
        return folder_paths.get_full_path(self.folder_name, filename)


    def get_full_path_or_raise(self, filename: str) -> str:
        """Returns the full path to a file in the directory, raising an error if the file does not exist."""
        return folder_paths.get_full_path_or_raise(self.folder_name, filename)


    def get_full_path_for_save(self, filename: str, overwrite: bool = False) -> str:
        """Returns the full path to save a file, ensuring it is unique if necessary.
        Args:
            filename (str): The name of the file.
            overwrite (bool, optional): If True, no unique name will be generated if the file already exists. Defaults to False.
        """
        assert not self._read_only, "Internal error: The directory was configured as read-only but a write operation was attempted."

        # be careful not to overwrite any pre-existing file
        # by appending a number to the filename if necessary
        counter = 1
        folder_path = folder_paths.get_folder_paths(self.folder_name)[0]
        name, ext   = os.path.splitext(filename)
        file_path   = os.path.join(folder_path, f"{name}{ext}")
        while not overwrite and os.path.exists(file_path):
            file_path = os.path.join(folder_path, f"{name}_{counter}{ext}")
            counter += 1
        return file_path


#----------------------- STANDARD COMFYUI DIRECTORY ------------------------#

class PredefinedDirectory(Directory):

    def __init__(self,
                 folder_name: str
                 ):
        super().__init__(folder_name,
                         custom     = False,
                         read_only  = True,
                         )


#-------------------------- CUSTOM MODEL DIRECTORY --------------------------#
class CustomModelDirectory(Directory):

    def __init__(self,
                folder_name         : str,
                supported_extensions: list = [".safetensors"]
                ):
        super().__init__(folder_name,
                         custom     = True,
                         read_only  = True,
                         parent_dir = folder_paths.models_dir,
                         supported_extensions = supported_extensions
                         )


#------------------------- CUSTOM OUTPUT DIRECTORY -------------------------#
class CustomOutputDirectory(Directory):

    def __init__(self,
                folder_name         : str,
                supported_extensions: list = [".safetensors"]
                ):
        super().__init__(folder_name,
                         custom     = True,
                         read_only  = False,
                         parent_dir = folder_paths.get_output_directory(),
                         supported_extensions = supported_extensions
                         )


VAE_DIR                = PredefinedDirectory("vae")
PIXART_CHECKPOINTS_DIR = CustomModelDirectory("pixart")
PIXART_LORAS_DIR       = CustomModelDirectory("pixart_loras")
T5_CHECKPOINTS_DIR     = CustomModelDirectory("t5")
PROMPT_EMBEDS_DIR      = CustomOutputDirectory("prompt_embeds")

