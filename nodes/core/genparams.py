"""
File    : gen_params.py
Purpose : Store and transport the parameters for image generation.
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
import re
import json
import struct
from datetime import datetime
_PLACEHOLDER = "$@"


def normalize_prefix(prefix: str) -> str:
    """Normalizes the given key prefix to ensure it is valid"""
    prefix = str(prefix if prefix is not None else "").strip()
    if prefix and not prefix.endswith('.'):
        return prefix + '.'
    return prefix


def _replace_template_placeholder(template: str, prompt: str) -> str:
    """Replaces placeholders in the given template with the provided prompt"""

    # if there is no placeholder,
    # return the prompt without any modifications
    if (not template) or (_PLACEHOLDER not in template):
        return prompt

    prefix, _, suffix = template.partition(_PLACEHOLDER)
    open, close       = prefix.rfind('{'), suffix.find('}')

    # if there is no placeholder brackets { },
    # return direct substitution of the placeholder
    # Example of this case:
    #   "A iPhone photo of $@ in a futuristic cityscape"
    if (open == -1 or close == -1):
        return f"{prefix}{prompt}{suffix}"

    # if there are closing brackets between the open brackets and the placeholder
    # return direct substitution of the placeholder
    # Example of this case:
    #   "A {device} photo of $@ in a futuristic {place}"
    if prefix.find('}', open) != -1 or suffix.rfind('{', close) != -1:
        return f"{prefix}{prompt}{suffix}"

    # beyond the placeholder, what is found between curly braces is the conditional content
    # - if there is prompt the conditional content appears
    # - if there is no prompt the conditional content does not appear
    if prompt:
        conditional_prefix = prefix[open+1:]
        conditional_suffix = suffix[:close]
        return f"{prefix[:open]}{conditional_prefix}{prompt}{conditional_suffix}{suffix[close+1:]}"
    else:
        return f"{prefix[:open]}{suffix[close+1:]}"



class GenParams(dict):
    """
    A class that represents the parameters for image generation.

    This class inherits from the built-in dict class, which means it can be used
    like a regular dictionary. It provides additional functionality for parsing
    and manipulating parameters.

    **Alternative Constructors:**
      - `from_raw_options(cls, raw_options)`: Creates a new GenParams object from the given raw options.
      - `from_safetensors_metadata(cls, file_path)`: Creates a new GenParams object from the given safetensors metadata.
      - `from_args(cls, args)`: Creates a new GenParams object from the given command arguments.
    """

    @classmethod
    def from_raw_options(cls, raw_options) -> "GenParams":
        """
        Creates a new GenParams object from the given raw options.

        This method is used to parse raw options from configuration files.
        Raw options are key-value pairs where both the key and value are strings.
        The values can be quoted, numeric, or boolean. If a value is quoted,
        it will remain as a string even if it looks like a number or boolean.
        """
        genparams = cls()
        for key, value in raw_options:
            if not isinstance(key,str) or not isinstance(value,str):
                continue

            # quotes force the value to remain as a string
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # if value is numeric, convert to int
            elif value.isdigit():
                value = int(value)
            # if value has a decimal point, convert to float
            elif '.' in value and value.replace('.', '', 1).isdigit():
                value = float(value)
            # if value is 'true' or 'false', convert to boolean
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'

            genparams[key] = value
        return genparams


    @classmethod
    def from_safetensors_metadata(cls, file_path: str) -> "GenParams":
        """
        Creates a new GenParams object by extracting metadata from the given safetensors file.

        The safetensors files store model weights but can also contain additional metadata
        that is useful for image generation. This method reads this metadata and creates a
        GenParams object with it.
        """
        # load the safetensors header from the file
        try:
            with open(file_path, "rb") as f:
                header_length = struct.unpack('<Q', f.read(8))[0]
                header_length = min(header_length, 10 * 1024 * 1024)
                header        = json.loads( f.read(header_length) )
        except Exception:
            header = {}

        # get the date the file was created
        try:
            file_date = datetime.fromtimestamp( os.path.getctime(file_path) )
        except Exception:
            file_date = datetime.now()

        # extract metadata from the header and inject file name and date
        metadata = header.get("__metadata__", {})
        metadata["file.name"] = os.path.basename(file_path)
        metadata["file.date"] = file_date.strftime("%Y-%m-%d")
        return cls(metadata)


    @classmethod
    def from_arguments(cls,
                       args: dict,
                       *,
                       template: "GenParams" = None
                       ) -> "GenParams":
        """
        Create a GenParams object from the given arguments.
        Args:
            args          (dict): A dictionary of argument names and their values.
            template (GenParams): A template GenParams object to be used as a base for the new GenParams object.
        """
        genparams = template.copy() if template else GenParams()

        # --prompt <text>
        # "base.prompt", "refiner.prompt"
        value = _get_str_value(args, "prompt")
        if value is not None:
            genparams.set_str("base.prompt"     , value, use_template=True)
            genparams.set_str("refiner.prompt"  , value, use_template=True)

        # --n, --no, --negative <text>
        # "base.negative", "refiner.negative"
        value = _get_str_value(args, "n", "no", "negative")
        if value is not None:
            genparams.set_str("base.negative"   , value, use_template=True)
            genparams.set_str("refiner.negative", value, use_template=True)

        # --c, --cfg <float>
        # "base.cfg"
        value, as_delta = _get_float_value(args, "c", "cfg")
        if value is not None:
            genparams.set_float("base.cfg", value, as_delta=as_delta)

        # --s, --seed <int>
        # "base.noise_seed"
        value, as_delta = _get_int_value(args, "s", "seed")
        if value is not None:
            genparams.set_int("base.noise_seed", value, as_delta=as_delta)

        # --v, --variant <int>
        # "refiner.noise_seed"
        value, as_delta = _get_int_value(args, "v", "variant")
        if value is not None:
            genparams.set_int("refiner.noise_seed", value, as_delta=as_delta)

        # --b, --batch <int>
        # "image.batch_size"
        value, as_delta = _get_int_value(args, "b", "batch")
        if value is not None:
            genparams.set_int("image.batch_size", value, as_delta=as_delta)

        return genparams


    def set_str(self, key: str, new_value: str, /, use_template: bool = False) -> None:
        """
        Stores a string value under the given key.
        The `use_template` parameter determines whether try to replace the placeholder in the current value.
        """
        if use_template:
            old_value = self.get(key)
            if isinstance(old_value, str):
                self[key] = _replace_template_placeholder( str(old_value), str(new_value) )
                return
        self[key] = str(new_value)


    def set_float(self, key: str, new_value: float, /, as_delta: bool = False):
        """
        Stores a floating-point value under the given key.
        The `as_delta` parameter can be used to add the new value to the existing one.
        """
        if as_delta:
            old_value = self.get(key)
            if isinstance(old_value, (int,float)):
                self[key] = float(old_value) + float(new_value)
                return
        self[key] = float(new_value)


    def set_int(self, key: str, new_value: int, /, as_delta: bool = False):
        """
        Stores an integer value under the given key.
        The `as_delta` parameter can be used to add the new value to the existing one.
        """
        if as_delta:
            old_value = self.get(key)
            if isinstance(old_value, (int,float)):
                self[key] = int(old_value) + int(new_value)
                return
        self[key] = int(new_value)


    def get_all_prefixed_keys(self, prefix: str) -> list[str]:
        """
        Returns a list of all keys with the given prefix in the GenParams object.
        If no keys are found, an empty list is returned.
        """
        if not isinstance(prefix,str):
            return []
        prefix = normalize_prefix(prefix)
        return [key for key in self.keys() if key.startswith(prefix)]


    def copy(self) -> "GenParams":
        """Returns a copy of the GenParams object."""
        return GenParams(self)


    def update(self,
               other: "GenParams",
               *,# keyword-only arguments #
               prefix_to_add: str = None
               ) -> None:
        """
        Updates the GenParams object with values from another GenParams object or dictionary.
        If a key is present in both objects, the value from the `other` object will overwrite it.
        The optional `prefix_to_add` parameter can be used to add a prefix to all keys before updating.
        """
        prefix_to_add = normalize_prefix(prefix_to_add)
        if not prefix_to_add:
            super().update(other)
            return
        for key, value in other.items():
            self[prefix_to_add+key] = value


    def update_from_group(self,
                          group_name: str,
                          *,# keyword-only arguments #
                          valid_subgroups: list[str] = None
                          ) -> None:
        """
        Updates the GenParams root with values from the specified internal group.
        The optional `valid_subgroups` parameter can be used to specify a list of valid subgroups for the group.
        """
        group_name     = normalize_prefix(group_name)
        group_name_len = len(group_name)

        if valid_subgroups:
            # when `valid_subgroups` is not empty,
            # the key prefix must match one of the formats: "<group_name>.<valid_subgroup_name>";
            # for example, if group_name is 'maingroup' and  valid_subgroups is ['sub1', 'sub2'],
            # then valid prefixes would be 'maingroup.sub1.' and 'maingroup.sub2.'
            valid_prefixes = [ f"{group_name}{normalize_prefix(subgroup_name)}" for subgroup_name in valid_subgroups ]
            group = {
                key[group_name_len:]: value
                for key,value in self.items()
                if any(key.startswith(valid_prefix) for valid_prefix in valid_prefixes)
                }
        else:
            # when `valid_subgroups` is empty,
            # the key prefix must match group_name.
            group = {
                key[group_name_len:]: value
                for key,value in self.items()
                if key.startswith(group_name) }

        # update the root with the group
        self.update(group)


    def __str__(self):
        """Return a string representation of the GenParams object."""
        return self.to_string(indent=4, width=90)


    def to_string(self, *, indent: int=4, width: int=-1) -> str:
        """Return a string representation of the GenParams object."""
        genparams = dict(self)
        string = "GenParams({\n"
        string += "# FILE\n"
        string += self.__pop_group_as_string("file."     , source=genparams, indent=indent, width=width)
        string += "# MODEL\n"
        string += self.__pop_group_as_string("modelspec.", source=genparams, indent=indent, width=width)
        string += "# IMAGE\n"
        string += self.__pop_group_as_string("image."    , source=genparams, indent=indent, width=width)
        string += "# SAMPLER\n"
        string += self.__pop_group_as_string("base."     , source=genparams, indent=indent, width=width)
        string += self.__pop_group_as_string("refiner."  , source=genparams, indent=indent, width=width)
        string += "# USER\n"
        string += self.__pop_group_as_string("user."     , source=genparams, indent=indent, width=width)
        string += "# STYLES\n"
        string += self.__pop_all_styles_as_string(         source=genparams, indent=indent             )
        string += "# OTHERS\n"
        string += self.__pop_group_as_string(""          , source=genparams, indent=indent, width=width)
        string += "})"
        return string

    @staticmethod
    def __pop_group_as_string(group_name: str, *, source: dict,
                              indent: int=4, width: int=-1
                              ) -> str:
        """Return a string representation of the specified group in the GenParams object."""
        width -= (indent + 22 + 2)
        group  = {key: value for key, value in source.items() if key.startswith(group_name)}
        string = ""
        for key, value in group.items():
            if isinstance(value, str):
                value = value.replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
                value = value[:width-3] + '...' if width>0 and len(value) > width else value
                value = f'"{value}"'
            string += f"{' ' * indent}{key:22}: {value},\n"
            del source[key]
        return string

    @staticmethod
    def __pop_all_styles_as_string(source: dict,
                                   indent: int=4) -> str:
        """Return a string representation of all styles in the GenParams object."""
        styles_names  = set()
        key_to_remove = []

        for key in source.keys():
            if key.startswith("styles."):
                style_name = key.split(".",2)[1]
                styles_names.add(style_name)
                key_to_remove.append(key)
        for key in key_to_remove:
            del source[key]

        string = ""
        for style_name in styles_names:
            key = f"styles.{style_name}.*"
            string += f"{' ' * indent}{key:22}: ...,\n"
        return string


#---------------------------- ARGUMENT PARSING -----------------------------#

def _get_str_value(args: dict, *keys) -> str:
    for key in keys:
        value = args.pop(key, None)
        if value is not None:
            return str(value)
    return None

def _get_int_value(args: dict, *keys) -> tuple[int | None, bool]:
    str_value = _get_str_value(args, *keys)
    if str_value is None:
        return None, False
    value, as_delta = _parse_int(str_value)
    if value is None:
        return None, False
    return value, as_delta

def _get_float_value(args: dict, *keys) -> tuple[float | None, bool]:
    str_value = _get_str_value(args, *keys)
    if str_value is None:
        return None, False
    value, as_delta = _parse_float(str_value)
    if value is None:
        return None, False
    return value, as_delta


def _parse_float(str_value: str,
                 *,
                 step     : float =  0.1,
                 min_value: float = -float('inf'),
                 max_value: float =  float('inf'),
                 ) -> tuple[float | None, bool]:
    """
    Parses a float value from a string with the format <number><brackets>.

    The number part is a standard float representation.
    The brackets part consists of '[' and ']' characters which increment
    and decrement the number by the given step, respectively.

    Args:
        str_value: The string to parse.
        step     : The increment/decrement value for each bracket.
        min_value: The minimum allowed value.
        max_value: The maximum allowed value.

    Returns:
        A tuple containing:
        - The parsed float value, or None if parsing fails.
        - A boolean indicating if the parsed value is a delta value.
    """
    if not str_value:
        return None, False

    # extract the number and the rest of the string
    match = re.match(r"^([-+]?\d*\.\d+?|)(.*)$", str_value.strip())
    if not match:
        return None, False

    number_str, remaining = match.groups()
    number   = float(number_str) if number_str else 0.0
    as_delta = not bool(number_str)

    # adjust the number based in the number of brackets in the remaining string
    for char in remaining.strip():
        if   char == "[":  number += step
        elif char == "]":  number -= step
        else:
            return None, False

    # if everything is OK,
    # return the number and a flag indicating if it's an offset
    number = max(min_value, min(number, max_value))
    return number, as_delta


def _parse_int(str_value: str,
               *,
               step       : int  =    1,
               min_value  : int  = None,
               max_value  : int  = None,
               should_wrap: bool = False,
               ) -> tuple[int | None, bool]:
    """
    Parses an integer value from a string with the format <number><brackets>.

    The number part is a standard integer representation.
    The brackets part consists of '[' and ']' characters which increment
    and decrement the number by the given step, respectively.

    Args:
        str_value  : The string to parse.
        step       : The increment/decrement value for each bracket.
        min_value  : The minimum allowed value.
        max_value  : The maximum allowed value.
        should_wrap: Whether to wrap around when reaching the minimum or maximum value.

    Returns:
        A tuple containing:
        - The parsed integer value, or None if parsing fails.
        - A boolean indicating if the parsed value is a delta value.
    """
    if not str_value:
        return None, False
    if min_value is None or max_value is None:
        should_wrap = False

    # extract the number and the rest of the string
    match = re.match(r"^([-+]?\d*)(.*)$", str_value.strip())
    if not match:
        return None, False

    number_str, remaining = match.groups()
    number   = int(number_str) if number_str else 0
    as_delta = not bool(number_str)

    # adjust the number based in the number of brackets in the remaining string
    for char in remaining.strip():
        if   char == "[":  number += int(step)
        elif char == "]":  number -= int(step)
        else:
            return None, False

    # ajust the value to be within min/max limits
    # if `should_wrap` is True then wrap around min/max limits
    if min_value is not None and number < min_value:
        number = (max_value + (number-min_value) + 1) if should_wrap else min_value
    if max_value is not None and number > max_value:
        number = (min_value + (number-max_value) - 1) if should_wrap else max_value

    # if everything is OK,
    # return the number and a flag indicating if it's an offset
    return number, as_delta

