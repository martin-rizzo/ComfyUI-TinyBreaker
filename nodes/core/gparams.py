"""
File    : gparams.py
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

def _is_prompt_key(key: str):
    """Returns True if the given key is used to specify prompts or negative prompts"""
    PROMPT_KEYS = (".prompt", ".negative")
    return f".{key}" in PROMPT_KEYS


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
    prefix_placeholder, suffix_placeholder = template.partition("prompt")

    # replace placeholder with prompt or empty string if no prompt is provided
    if prompt:
        return f"{prefix}{prefix_placeholder}{prompt}{suffix_placeholder}{suffix}"
    else:
        return f"{prefix}{suffix}"


class GParams(dict):

    @classmethod
    def from_raw_options(cls, options) -> "GParams":
        """
        Creates a new GParams object from the given raw options.

        This method is used to parse raw options from configuration files.
        Raw options are key-value pairs where both the key and value are strings.
        The values can be quoted, numeric, or boolean. If a value is quoted,
        it will remain as a string even if it looks like a number or boolean.
        """
        gparams = cls()
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

            gparams[key] = value
        return gparams



    @classmethod
    def from_template_and_values(cls, template: "GParams", values: "GParams") -> "GParams":
        """
        Creates a new GParams object by merging the template and values.

        If the template contains placeholders for prompts or negative prompts,
        they are replaced with the corresponding values from the values dictionary.
        The rest of the values are added directly to the new GParams object.
        """
        gparams = cls(template)

        for key, value in values.items():
            # if the key is a prompt key,
            # then replace the template placeholder with the provided value
            if (key in template) and _is_prompt_key(key):
                gparams[key] = _replace_template_placeholder( template[key], value )
            else:
                gparams[key] = value

        return gparams


    def to_text(self) -> str:
        """
        Converts the GParams dictionary back into a text format.
        """
        lines = []
        for key, value in self.items():
            lines.append(f"{key}={value}")
        return '\n'.join(lines)

    def __str__(self):
        return self.to_text()


