"""
File    : styles.py
Purpose : A class for easily managing a set of pre-configured styles
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 20, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import configparser

class Styles:
    """
    A class for managing a set of pre-configured styles.
    """
    def __init__(self, styles_dict: dict = None, /):
        self.styles = styles_dict if styles_dict else {}


    def __len__(self):
        """Returns the number of styles in this instance."""
        return len(self.styles)


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

        # this trick ensures the parser treats the unnamed first section
        # as "__default__" for use as the base when no template is specified
        if "[__default__]" not in config_string:
            config_string = "[__default__]\n" + config_string

        # use `ConfigParser` to parse the string,
        # each section is a style with its properties
        config = configparser.ConfigParser(empty_lines_in_values=False, default_section="")
        config.read_string(config_string)
        styles = cls()
        for section in config.sections():

            # section names are expected in the format "[style:template]" where template is optional
            # if a template is specified, that style will be used as the base
            style_name, _, template_name = section.partition(":")
            style_name,    template_name = style_name.strip(), template_name.strip()
            styles.add_style(style_name, config.items(section), template_name = template_name or "__default__")

        styles.remove_style("__default__")
        return styles


    @classmethod
    def from_merge(cls, base_styles: "Styles", new_styles: "Styles") -> "Styles":
        """
        Returns a new instance of Styles populated with all styles defined in both base_styles and new_styles.

        If a style is defined in both base_styles and new_styles, the value from `new_styles` takes precedence.
        The merge performs a shallow copy so that references to each style are maintained.

        Args:
            base_styles (Styles): The base Styles instance.
            new_styles  (Styles): The new Styles instance.
        """
        merged_styles = base_styles.styles.copy()
        merged_styles.update(new_styles.styles)
        return cls(merged_styles)


    @classmethod
    def from_single_style(cls, style_name: str, style_raw_kv_params: dict) -> "Styles":
        """
        Returns a new instance of Styles populated with a single style defined by the given name and parameters.
        Args:
            style_name         : The name of the style to be added.
            style_raw_kv_params: A dictionary containing key-value pairs for the style parameters.
        """
        styles = cls()
        styles.add_style(style_name, style_raw_kv_params)
        return styles


    def add_style(self,
                  style_name         : str,
                  style_raw_kv_params: dict,
                  /,*,
                  template_name: str = None
                  ):
        """
        Adds a new style from a raw dictionary of key-value pairs.
        Args:
            style_name              : The name of the style to be added.
            style_raw_kv_params     : A dictionary containing key-value pairs for the style parameters.
            template_name (optional): The name of an existing style to use as a template.
        """
        from .genparams import GenParams
        if style_name in self.styles:
            raise ValueError(f"Style '{style_name}' already exists.")

        genparams = GenParams.from_raw_kv_params(style_raw_kv_params)
        template  = self.styles.get(template_name) if template_name else None
        if template:
            _combined = template.copy()
            _combined.update(genparams)
            genparams = _combined

        self.styles[style_name] = genparams


    def remove_style(self, style_name):
        """Removes a style from the dictionary."""
        if style_name not in self.styles:
            raise ValueError(f"Style '{style_name}' does not exist.")
        del self.styles[style_name]


    def update(self, new_styles: "Styles", /, overwrite: bool = True):
        """
        Updates this instance with styles from another Styles object.
        Args:
            new_styles (Styles): The Styles object to update from.
            overwrite    (bool): If False, only new styles will be added.
                                 If True (default), existing styles will be overwritten.
        """
        if overwrite:
            self.styles.update(new_styles.styles)
            return
        for name in new_styles.names():
            if name not in self.styles:
                self.styles[name] = new_styles.styles[name]


    def get_genparams_for_style(self, style_name):
        """Returns the generation parameters for a given style or None if not found."""
        return self.styles.get(style_name)


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

def _merge_new_styles_preserving_existing(old_styles_by_version: dict[str,Styles],
                                          new_styles           : Styles
                                          ) -> None:
    """
    Updates older versions of styles with new styles but without overwriting existing ones.
    """
    for old_styles in old_styles_by_version.values():
        old_styles.update( new_styles, overwrite=False )


def _apply_style_order(style_names_to_sort: list[str],
                       *,
                       dir_path      : str,
                       order_filename: str = "order.conf",
                       ) -> list[str]:
    """
    Applies the order defined in a configuration file to a list of style names.
    Args:
        style_names_to_sort (list): The list of style names to be ordered.
        dir_path             (str): The directory path containing the order configuration file.
        order_filename       (str): The name of the order configuration file. Defaults to "order.conf".
    """
    # read the style order from file (if it exists)
    name_order = []
    try:
        with open(os.path.join(dir_path, order_filename), 'r') as f:
            for line in f:
                line = line.strip()
                if line and not (line.startswith('#') or line.startswith(';')):
                    name_order.extend([s.strip() for s in line.split(',')])
    except FileNotFoundError:
        name_order = []

    # apply the style order to `style_names_to_sort`
    sorted_names   = []
    unsorted_names = list(style_names_to_sort)
    for name in name_order:
        if name in style_names_to_sort:
            unsorted_names.remove(name)
            sorted_names.append(name)
    return sorted_names + unsorted_names


def load_all_styles_versions(dir_path       : str,
                             file_prefix    : str = "styles",
                             extension      : str = ".ini",
                             order_filename : str = "order.conf",
                             ) -> tuple[ dict[str,Styles], list[str], list[str] ]:
    """
    Loads multiple Styles object from files in the specified directory.
    Args:
        dir_path    (str): The path to the directory containing style files.
        file_prefix (str): The prefix of the style files. Default is "styles".
        extension   (str): The extension of the style files. Default is ".cfg".
    Returns:
        A tuple containing:
        - A dictionary mapping style version to their respective Styles objects.
        - A list of all versions found in the directory sorted by new version first.
        - A list of style names found across all versions.
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
    # and sort by the number/version that follows the prefix in the filename (the oldest version first)
    sorted_file_names = [f for f in os.listdir(dir_path) if f.startswith(file_prefix) and f.endswith(extension)]
    sorted_file_names.sort( key=lambda name: _extract_number(name, file_prefix_len, extension_len) )

    styles_by_version      = {}
    chronological_versions = []
    new_version_styles     = Styles()

    # load the styles, indexing them by their version number
    for file_name in sorted_file_names:
        prev_version_styles = new_version_styles

        version_number = _extract_number(file_name, file_prefix_len, extension_len)
        version        = f"{version_number//10}.{version_number%10}" if version_number > 0 else "???"
        file_path      = os.path.join(dir_path, file_name)

        # merge the styles from the file with the ones from previous version,
        # if any new style was added then add it to all previous versions as well
        new_version_styles = Styles.from_merge( prev_version_styles, Styles.from_file(file_path) )
        if len(new_version_styles) > len(prev_version_styles):
            _merge_new_styles_preserving_existing( styles_by_version, new_version_styles )

        # add the loaded styles to the dictionary
        styles_by_version[version] = new_version_styles
        chronological_versions.append(version)

    # sort the style names by their order in the `order.conf` file
    style_names = _apply_style_order(new_version_styles.names(),
                                     dir_path       = dir_path,
                                     order_filename = order_filename
                                     )
    return styles_by_version, list(reversed(chronological_versions)), style_names



