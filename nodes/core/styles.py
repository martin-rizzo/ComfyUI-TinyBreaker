"""
File    : style.py
Purpose : Provides functionality to define and apply custom styles to prompts.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 18, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import torch

class _TypedStyle:
    """
    A class that represents a custom style associated with a type of prompt.
    """

    def __init__(self,
                 type           : str,
                 prefix         : str = "",
                 suffix         : str = "",
                 prefix_optional: str = "",
                 suffix_optional: str = "",
                 prefix_tensor  : torch.Tensor = None,
                 suffix_tensor  : torch.Tensor = None):
        """
        Initializes a new instance of the _TypedStyle class.
        Arguments:
            type           : The type of prompt associated with this style (eg. "PROMPT" "NEGATIVE_PROMPT", "REFINER", etc)
            prefix         : A string that will be prepended to the prompt.
            suffix         : A string that will be appended to the prompt.
            prefix_optional: An optional string that will be prepended to the prompt if it is not empty.
            suffix_optional: An optional string that will be appended to the prompt if it is not empty.
            prefix_tensor  : A torch.Tensor that will be prepended to the prompt after it has coded into embedding
            suffix_tensor  : A torch.Tensor that will be appended to the prompt after it has coded into embedding
        """
        self.type            = type.strip().upper()
        self.prefix          = prefix
        self.suffix          = suffix
        self.prefix_optional = prefix_optional
        self.suffix_optional = suffix_optional
        self.prefix_tensor   = prefix_tensor
        self.suffix_tensor   = suffix_tensor

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        Returns a string representation of the _TypedStyle object.
        """
        return f"_TypedStyle(type={self.type}, prefix='{self.prefix}', suffix='{self.suffix}', prefix_optional='{self.prefix_optional}', suffix_optional='{self.suffix_optional}')"


    def apply_to_text(self, text: str) -> str:
        """
        Applies the custom style to a given prompt text.
        """
        text = text.strip()
        if not text:
            return f"{self.prefix}{self.suffix}"
        else:
            return f"{self.prefix}{self.prefix_optional}{text}{self.suffix_optional}{self.suffix}"
    

    def apply_to_tensor(self, tensor: torch.Tensor, style_strength: float = 1.0) -> torch.Tensor:
        """
        Applies the custom style to a given tensor.
        """
        # concatenate the prefix and suffix tensors with the input tensor along the appropriate dimension
        if self.prefix_tensor is not None:
            tensor = torch.cat((self.prefix_tensor * style_strength, tensor), dim=0)
        if self.suffix_tensor is not None:
            tensor = torch.cat((tensor, self.suffix_tensor * style_strength), dim=0)

        # return the modified tensor
        return tensor


    @classmethod
    def from_text_line(cls, line: str) -> "_TypedStyle":
        """
        Creates a _TypedStyle object from a text line.
        """

        # check if the input line is empty or starts with a comment character
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        type, content = line.split(':', 1)
        type    = type.strip()
        content = content.strip()

        # remove any surrounding quotes if present in the content string
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]

        typed_style = cls.from_content(type, content)
        return typed_style


    @classmethod
    def from_content(cls, type: str, content: str) -> "_TypedStyle":
        """
        Creates a _TypedStyle object from a name and content string.
        """
        content = content.strip()

        # if the content is empty, return an instance with default values
        if not content:
            return cls(type)

        # if the content does not contain any placeholders, add it as default to the end
        if '{' not in content:
            content = content + "{. prompt}"

        # split the content into 3 parts: prefix, prompt and suffix
        prefix,  content = content.split('{', 1)
        content, suffix  = content.split('}', 1)

        # prefix_optional is everything before the string "prompt"
        # suffix_optional is everything after the string "prompt"
        prefix_optional, suffix_optional = content.split("prompt", 1)

        return cls(type,
                   prefix=prefix,
                   suffix=suffix,
                   prefix_optional=prefix_optional,
                   suffix_optional=suffix_optional
                   )


#---------------------------------- STYLE ----------------------------------#
class Style(dict):
    
    def apply_to_text(self, text: str, type: str) -> str:
        """
        Applies the style to a given text.
        """
        # get the custom style object for the given type
        type_style = self.get(type.upper())

        # if no custom style is found, return the original text
        if not type_style:
            return text
        # apply the custom style to the text and return the modified text
        return type_style.apply_to_text(text)


    def apply_to_tensor(self, tensor: torch.Tensor, style_strength: float, type: str) -> torch.Tensor:
        """
        Applies the style to a given tensor.
        """
        # get the custom style object for the given type
        type_style = self.get(type.upper())

        # if no custom style is found, return the original tensor
        if not type_style:
            return tensor

        # apply the custom style to the tensor and return the modified tensor
        return type_style.apply_to_tensor(tensor, style_strength)


    @classmethod
    def from_text(cls, text: str) -> "Style":
        """
        Creates a Style object from a string of style definitions.
        """
        style = cls()
        for text_line in text.split('\n'):
            typed_style = _TypedStyle.from_text_line(text_line)
            if typed_style:
                style[typed_style.type] = typed_style
        return style


    @classmethod
    def from_file(cls, file_path: str) -> "Style":
        """
        Creates a Style object from a file containing style definitions.
        """
        with open(file_path, 'r') as f:
            text = f.read()
        return cls.from_text(text)
    

#---------------------------- STYLE COLLECTION -----------------------------#
class StyleCollection(dict):

    @classmethod
    def from_directory(cls, directory: str) -> "StyleCollection":
        """
        Creates a StyleCollection object from a directory of files containing style definitions.
        """
        style_collection = cls()

        for file_name in os.listdir(directory):
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(directory, file_name)
                style     = Style.from_file(file_path)
                if style:
                    style_name = cls._generate_style_name(file_name)
                    # print("##>> adding style:", style_name)
                    style_collection[style_name] = style

        return style_collection


    @staticmethod
    def _generate_style_name(file_name: str):
        name = os.path.splitext(file_name)[0]
        return name.replace('_', ' ')



