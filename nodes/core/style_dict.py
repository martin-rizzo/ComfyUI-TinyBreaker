"""
File    : style_dict.py
Purpose : Manages a dictionary of image styles to apply to user prompts.
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
import configparser
from .gen_params import GenParams

class StyleDict:
    """
    A class to manage a dictionary of styles.
    """
    def __init__(self):
        self.styles = {}


    @classmethod
    def from_file(cls, file_path: str) -> 'StyleDict':
        """
        Returns a new instance of StyleDict populated with all styles defined in the file.
        Args:
            file_path (str): The path to the configuration file.
        """
        content = ""
        with open(file_path, 'r') as file:
            content = file.read()
        return cls.from_string(content)


    @classmethod
    def from_string(cls, config_string: str) -> 'StyleDict':
        """
        Returns a new instance of StyleDict populated with all styles defined in the string.
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
        self.styles[style_name] = GenParams.from_raw_options(style_raw_options)


    def get_style_genparams(self, style_name):
        """Returns the generation parameters for a given style or an empty `GenParams` if the style is not found."""
        return self.styles.get(style_name) or GenParams()


    def remove_style(self, style_name):
        """Removes a style from the dictionary."""
        if style_name not in self.styles:
            raise ValueError(f"Style '{style_name}' does not exist.")
        del self.styles[style_name]


    def names(self):
        """Returns a list of all style names."""
        return list(self.styles.keys())


