"""
File    : styles.py
Purpose : A class for easily managing a set of pre-configured styles
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 20, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import configparser
from .genparams import GenParams


class Styles:
    """
    A class for managing a set of pre-configured styles.
    """
    def __init__(self):
        self.styles = {}


    @classmethod
    def from_file(cls, file_path: str) -> "Styles":
        """
        Returns a new instance of Styles populated with all styles defined in the file.
        Args:
            file_path (str): The path to the configuration file.
        """
        content = ""
        with open(file_path, 'r') as file:
            content = file.read()
        return cls.from_string(content)


    @classmethod
    def from_string(cls, config_string: str) -> "Styles":
        """
        Returns a new instance of Styles populated with all styles defined in the string.
        Args:
            config_string (str): A string containing configuration data.
        """
        # ensure config has a [DEFAULT] section, this trick ensures that
        # the parser treats the unnamed first section as a default section
        if "[DEFAULT]" not in config_string:
            config_string = "[DEFAULT]\n" + config_string

        # use configparser to extract any defined styles
        config = configparser.ConfigParser(empty_lines_in_values=False)
        config.read_string(config_string)
        styles = cls()
        for section in config.sections():
            styles.add_style(section, config.items(section))

        return styles


    def add_style(self,
                  style_name       : str,
                  style_raw_options: tuple[str,str]
                  ):
        """Adds a new style to the dictionary"""
        if style_name in self.styles:
            raise ValueError(f"Style '{style_name}' already exists.")
        self.styles[style_name] = GenParams.from_raw_options(style_raw_options)


    def remove_style(self, style_name):
        """Removes a style from the dictionary."""
        if style_name not in self.styles:
            raise ValueError(f"Style '{style_name}' does not exist.")
        del self.styles[style_name]


    def get_genparams(self, style_name):
        """Returns the generation parameters for a given style or an empty `GenParams` if the style is not found."""
        return self.styles.get(style_name) or GenParams()


    def names(self):
        """
        Returns a dynamic view of the names of all contained styles.

        This method returns a view object, similar to `dict.keys()`, that reflects
        the current set of style names. Changes to the styles will be immediately
        visible through this view.
        To obtain a static list of style names, use `list( styles.names() )`.
        """
        return self.styles.keys()


#---------------------------------------------------------------------------#

def load_all_styles_versions(dir_path   : str,
                             file_prefix: str="styles",
                             extension  : str=".ini"
                             )-> tuple[ dict[str,Styles], str ]:
    """
    Loads multiple Styles object from files in the specified directory.
    Args:
        dir_path    (str): The path to the directory containing style files.
        file_prefix (str): The prefix of the style files. Default is "styles".
        extension   (str): The extension of the style files. Default is ".cfg".
    Returns:
        A dictionary mapping style version to their respective Styles objects.
    """

    def _extract_number(string:str, prefix_len, suffix_len) -> int:
        """Extracts a number from a string"""
        number = string[prefix_len:-suffix_len] if suffix_len else string[prefix_len:]
        number = number.lstrip('_').lstrip('v') # remove leading 'v' for version strings
        if not number.isdigit(): return 0
        return int(number)

    file_prefix_len = len(file_prefix)
    extension_len   = len(extension)

    # find all files in the directory that match the specified prefix and extension
    # and sort by the number/version that follows the prefix in the filename (first the highest value ones)
    sorted_file_names = [f for f in os.listdir(dir_path) if f.startswith(file_prefix) and f.endswith(extension)]
    sorted_file_names.sort( key=lambda name: -_extract_number(name, file_prefix_len, extension_len) )

    last_version = ""
    styles_by_version = {}

    # load the styles, indexing them by their version number
    for file_name in sorted_file_names:
        version_number = _extract_number(file_name, file_prefix_len, extension_len)
        version   = f"{version_number//10}.{version_number%10}" if version_number > 0 else "???"
        file_path = os.path.join(dir_path, file_name)
        styles_by_version[version] = Styles.from_file(file_path)
        if not last_version:
            last_version = version

    return styles_by_version, last_version


