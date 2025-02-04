"""
File    : directories.py
Purpose : Defines directory paths as constants for various model types.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 The directory objects can be used as follows:

      path      = VAE_DIR.folder_name
      filenames = VAE_DIR.get_filename_list()
      VAE_DIR.get_full_path(filename)
      VAE_DIR.get_full_path_or_raise(filename)
      VAE_DIR.get_full_path_for_save(filename, overwrite=False)

"""
import os
import folder_paths

#---------------------------------------------------------------------------#
class _ProjectDirectory:

    def __init__(self,
                 folder_name : str
                 ):
        _utils_dir   = os.path.dirname(__file__)
        _nodes_dir   = os.path.dirname(_utils_dir)
        _project_dir = os.path.dirname(_nodes_dir)
        self.folder_name = folder_name
        self.paths       = (os.path.join(_project_dir, folder_name), )


    def get_full_path(self, filename: str) -> str:
        """Returns the full path to a file in the directory, returning None if the file is not found."""
        for path in self.paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path
        return None


#---------------------------------------------------------------------------#
class _ComfyDirectory:
    """
    A base class to manage ComfyUI directories for storing and accessing files and models.
    This class will be subclassed to create specific directories for different purposes.
    """

    def __init__(self,
                 folder_name : str,
                 custom      : bool = False,
                 read_only   : bool = False,
                 parent_dir  : str  = None,
                 supported_extensions: list = None,
                 ):
        """Initializes a comfy directory.
        Args:
            folder_name    (str): The name of the directory.
            custom        (bool): If True, the directory is a custom directory created for this project.
                                  If False, the directory is a standard directory provided by ComfyUI.
            read_only     (bool): If True, the directory would be considered read-only.
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

        if self._custom:
            # this code handles the initialization of custom directories:
            #  - it adds the directory to ComfyUI's internal list of directories
            #  - it also creates the directory if it doesn't already exist
            if not self._initialized:
                parent_dir           = self._parent_dir or folder_paths.models_dir
                supported_extensions = self._supported_extensions or [".safetensors", ".sft"]
                full_path  = os.path.join(parent_dir, self._folder_name)
                folder_paths.folder_names_and_paths[self._folder_name] = (
                    [full_path],
                    set(supported_extensions)
                    )
                os.makedirs(full_path, exist_ok=True)

        self._initialized = True
        return self._folder_name


    @property
    def paths(self):
        """Returns the list of paths associated with this directory."""
        return folder_paths.get_folder_paths(self.folder_name)


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


#---------------------------------------------------------------------------#
class _ComfyModelDirectory(_ComfyDirectory):

    def __init__(self,
                folder_name : str,
                custom      : bool = False,
                supported_extensions: list = [".safetensors"]
                ):
        super().__init__(folder_name,
                         custom     = custom,
                         read_only  = True,
                         parent_dir = folder_paths.models_dir,
                         supported_extensions = supported_extensions
                         )


#---------------------------------------------------------------------------#
class _ComfyOutputDirectory(_ComfyDirectory):

    def __init__(self,
                folder_name : str,
                custom      : bool = False,
                supported_extensions: list = [".safetensors"]
                ):
        super().__init__(folder_name,
                         custom     = custom,
                         read_only  = False,
                         parent_dir = folder_paths.get_output_directory(),
                         supported_extensions = supported_extensions
                         )


#===========================================================================#
#//////////////////////////// DIRECTORY OBJECTS ////////////////////////////#
#===========================================================================#

PROJECT_DIR                 = _ProjectDirectory(".")
STYLES_DIR                  = _ProjectDirectory("styles")
CHECKPOINTS_DIR             = _ComfyModelDirectory("checkpoints")
VAE_DIR                     = _ComfyModelDirectory("vae")
TRANSCODERS_DIR             = VAE_DIR
TINYBREAKER_CHECKPOINTS_DIR = CHECKPOINTS_DIR
#TINYBREAKER_CHECKPOINTS_DIR = _ComfyModelDirectory("tinybreaker", custom=True)

