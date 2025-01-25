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
import json
import struct
_PLACEHOLDER = "$@"


def normalize_prefix(prefix: str) -> str:
    """Normalizes the given key prefix to ensure it is valid"""

    # remove whitespaces and ensure that the prefix ends with a period '.' 
    prefix = prefix.strip()
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
      - `from_safetensors_metadata(cls, path)`: Creates a new GenParams object from the given safetensors metadata.
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
    def from_safetensors_metadata(cls, path: str) -> "GenParams":
        """
        Creates a new GenParams object by extracting metadata from the given safetensors file.

        The safetensors files store model weights but can also contain additional metadata
        that is useful for image generation. This method reads this metadata and creates a
        GenParams object with it.
        """
        # load the safetensors header from the file
        try:
            with open(path, "rb") as f:
                header_length = struct.unpack('<Q', f.read(8))[0]
                header_length = min(header_length, 10 * 1024 * 1024)
                header        = json.loads( f.read(header_length) )
        except Exception:
            header = {}

        # extract metadata from the header
        metadata = header.get("__metadata__", {})
        metadata["filename"] = os.path.basename(path)
        return cls(metadata)


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


    def __str__(self):
        """Return a string representation of the GenParams object."""
        string = "GenParams(\n"
        for key,value in self.items():
            if isinstance(value, str):
                value = value.replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
                value = value[:96] + '...' if len(value) > 130 else value
                value = f'"{value}"'
            string += f"    {key:16}: {value}\n"
        string += ")"
        return string

