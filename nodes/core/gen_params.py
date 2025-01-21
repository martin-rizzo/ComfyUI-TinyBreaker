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

def _is_prompt(key: str):
    """Returns True if the given key is used to specify prompts or negative prompts"""
    PROMPT_KEYS = (".prompt", ".negative")
    return f".{key}".endswith(PROMPT_KEYS)

def _normalize_prefix(prefix: str) -> str:
    """Normalizes the given prefix to ensure it is valid"""
    # remove whitespaces and ensure that the prefix ends with a period '.' 
    prefix = prefix.strip()
    if prefix and not prefix.endswith('.'):
        return prefix + '.'
    return prefix

def _build_key(prefix: str, var_name: str) -> str:
    """Returns the key by combining the prefix and variable name"""
    prefix = _normalize_prefix(prefix)
    return prefix + var_name.strip()


def _replace_template_placeholder(template: str, prompt: str) -> str:
    """Replaces placeholders in the given template with the provided prompt"""

    # if the template is empty, return the prompt without any modifications
    if not template:
        return prompt

    # if there is no placeholder, the template acts as a prefix to the prompt
    if '{' not in template:
        return f"{template}. {prompt}" if prompt else template

    # parse template into prefix, placeholder, and suffix
    prefix  , _, template = template.partition('{')
    template, _, suffix   = template.partition('}')
    prefix_placeholder, _, suffix_placeholder = template.partition("prompt")

    # replace placeholder with prompt or empty string if no prompt is provided
    if prompt:
        return f"{prefix}{prefix_placeholder}{prompt}{suffix_placeholder}{suffix}"
    else:
        return f"{prefix}{suffix}"


class GenParams(dict):

    @classmethod
    def from_raw_options(cls, options) -> "GenParams":
        """
        Creates a new GenParams object from the given raw options.

        This method is used to parse raw options from configuration files.
        Raw options are key-value pairs where both the key and value are strings.
        The values can be quoted, numeric, or boolean. If a value is quoted,
        it will remain as a string even if it looks like a number or boolean.
        """
        genparams = cls()
        for key, value in options:
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
    def from_template_and_data(cls, template: "GenParams", data: "GenParams") -> "GenParams":
        """
        Creates a new GenParams object by merging the template and data parameters.

        If the `template` contains prompts with placeholders, these placeholders
        are replaced with the corresponding values from the `data` parameters.

        The rest of values that are not prompts are added directly to the
        new GenParams object, having as priority those from the `data` parameters.

        """
        genparams = cls(data)
        for template_key, template_value in template.items():

            # if the template value is a prompt,
            # then replace the template placeholder with the provided value
            if _is_prompt(template_key):
                value = genparams.get(template_key, "")
                genparams[template_key] = _replace_template_placeholder( template_value, value )

            # any template value not found in `data` is added as it is
            elif template_key not in genparams:
                genparams[template_key] = template_value

        return genparams


    def set(self, key: str, value, /) -> None:
        """
        Sets the value of a key in the GenParams object.
        If the key already exists, it is replaced with the new value.
        """
        self[key] = value


    def add(self, key: str, value, /) -> None:
        """
        Mathematically adds a value to an existing key in the GenParams object.
        If the key does not exist, it is created with the given value.
        """
        # force `value` to be a numeric value
        if isinstance(value,str):
            value = float(value) if value.replace('.', '', 1).isdigit() else 0
        else:
            value = float(value)

        # update existing keys or add new ones
        if key in self:
            self[key] += value
        else:
            self[key] = value



    def set_prefixed(self, prefix: str, var_name: str, value, /) -> None:
        """
        Sets the value of a variable with a given prefix in the GenParams object.
        If the key already exists, it is replaced with the new value.
        """
        if not isinstance(prefix,str) or not isinstance(var_name,str):
            return
        self[ _build_key(prefix,var_name) ] = value


    def get_prefixed(self, prefix: str, var_name: str, default = None, /):
        """
        Retrieves the variable of a key with a given prefix from the GenParams object.
        If the key does not exist, the default value is returned.
        """
        if not isinstance(prefix,str) or not isinstance(var_name,str):
            return default
        return self.get( _build_key(prefix,var_name), default )


    def pop_prefixed(self, prefix: str, var_name: str, default = None, /):
        """
        Removes and returns the value of a variable with a given prefix from the GenParams object.
        If the key does not exist, the default value is returned.
        """
        if not isinstance(prefix,str) or not isinstance(var_name,str):
            return default
        return self.pop( _build_key(prefix,var_name), default )


    def get_all_prefixed_keys(self, prefix: str) -> list[str]:
        """
        Returns a list of all keys with the given prefix in the GenParams object.
        If no keys are found, an empty list is returned.
        """
        if not isinstance(prefix,str):
            return []
        prefix = _normalize_prefix(prefix)
        return [key for key in self.keys() if key.startswith(prefix)]


    def copy(self) -> "GenParams":
        """Returns a copy of the GenParams object."""
        return GenParams(self)


    def to_text(self) -> str:
        """
        Converts the GenParams dictionary back into a text format.
        """
        lines = []
        for key, value in self.items():
            lines.append(f"{key}={value}")
        return '\n'.join(lines)


    def __str__(self):
        return self.to_text()


