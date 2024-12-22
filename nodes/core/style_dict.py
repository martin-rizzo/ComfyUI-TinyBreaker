"""
File    : style_dict.py
Purpose : 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 20, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import configparser
from .gparams import GParams

class StyleDict:
    """
    A class to manage a dictionary of styles.
    """
    def __init__(self):
        self.styles = {}


    @classmethod
    def from_file(cls, file_path: str) -> 'StyleDict':
        """
        Creates an instance of StyleDict from a configuration file.
        Args:
            file_path (str): The path to the configuration file.
        Returns:
            An instance of StyleDict populated with all styles defined in the file.
        """

        # read the configuration file content into a string
        # but ensure it has a [DEFAULT] section, this trick ensures that
        # configparser treats the unnamed section as a default section
        with open(file_path, 'r') as file:
            config_content = file.read()
        if "[DEFAULT]" not in config_content:
            config_content = "[DEFAULT]\n" + config_content

        config = configparser.ConfigParser(empty_lines_in_values=False)
        config.read_string(config_content)
        style_dict = cls()
        for section in config.sections():
            style_dict.add_new_style(section, config.items(section))

        return style_dict


    def add_new_style(self,
                      style_name       : str,
                      style_raw_options: tuple[str,str]
                      ):
        """Adds a new style to the dictionary"""
        if style_name in self.styles:
            raise ValueError(f"Style '{style_name}' already exists.")
        self.styles[style_name] = GParams.from_raw_options(style_raw_options)


    def get_style_params(self, style_name):
        """Returns the parameters for a given style or an empty `GParams` if the style is not found."""
        return self.styles.get(style_name) or GParams()


    def remove_style(self, style_name):
        """Removes a style from the dictionary."""
        if style_name not in self.styles:
            raise ValueError(f"Style '{style_name}' does not exist.")
        del self.styles[style_name]


    def list_styles(self):
        """Returns a list of all styles in the dictionary."""
        return list(self.styles.keys())


