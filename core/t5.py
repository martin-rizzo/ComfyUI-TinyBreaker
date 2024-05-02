"""
  File    : p5.py
  Brief   : t5 tokenizer y encoder implementados con transformers
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : Apr 29, 2024
  Repo    : https://github.com/martin-rizzo/ComfyUI-PixArt
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      PixArt Node Collection for ComfyUI
             Nodes providing support for PixArt models in ComfyUI

     Copyright (c) 2024 Martin Rizzo

     Permission is hereby granted, free of charge, to any person obtaining
     a copy of this software and associated documentation files (the
     "Software"), to deal in the Software without restriction, including
     without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software, and to
     permit persons to whom the Software is furnished to do so, subject to
     the following conditions:

     The above copyright notice and this permission notice shall be
     included in all copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

DOCUMENTATION
=============

 Loading a checkpoint reducing compute and memory as much as possible:
 - https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html

 T5v1.1: T5v1.1 is an improved version of T5 with some architectural tweaks:
 - https://huggingface.co/docs/transformers/en/model_doc/t5v1.1

"""
import os
import gc
import torch
from typing      import Optional, Union, Dict, List
from safetensors import safe_open

from transformers import               \
    T5Config       as HF_T5Config,     \
    T5Tokenizer    as HF_T5Tokenizer,  \
    T5EncoderModel as HF_T5EncoderModel

from comfy.sd1_clip import                         \
    load_embed         as comfy_load_embed,        \
    token_weights      as comfy_token_weights,     \
    escape_important   as comfy_escape_important,  \
    unescape_important as comfy_unescape_important \


# Google T5 v1.1 models
# =====================
#
#          | #Params | #Params   |
#   Models | encoder | enco+deco | layers | d^model |  d^ff | d^kv | #heads
# ---------+---------+-----------+--------+---------+-------+------+-------
#   small  |    ?    |     77M   |    8   |    512  |  1024 |  64  |   6
#    base  |    ?    |    250M   |   12   |    768  |  2048 |  64  |  12
#   large  |    ?    |    780M   |   24   |   1024  |  2816 |  64  |  16
#      xl  |    ?    |     3B    |   24   |   2048  |  5120 |  64  |  32
#     xxl  |  4.7B   |    11B    |   24   |   4096  | 10240 |  64  |  64

T5PredefinedConfigs={
    'xxl': {
            'vocab_size': 32128,
            'd_model': 4096,
            'd_kv': 64,
            'd_ff': 10240,
            'num_layers': 24,
            'num_decoder_layers': 24,
            'num_heads': 64,
            'relative_attention_num_buckets': 32,
            'relative_attention_max_distance': 128,
            'dropout_rate': 0.1,
            'layer_norm_epsilon': 1e-6,
            'initializer_factor': 1.0,
            'feed_forward_proj': "gated-gelu",
            'is_encoder_decoder': True,
            'use_cache': True,
            'pad_token_id': 0,
            'eos_token_id': 1,
            #'classifier_dropout': 0.0
        }
    }

#=============================== T5Tokenizer ===============================#
class T5Tokenizer:

    def __init__(self,
                 tokenizer,
                 max_length,
                 embedding_dir,
                 embedding_key,
                 embedding_size
                 ):

        # tokenizer
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # embeddings
        self.embedding_dir  = embedding_dir
        self.embedding_tag  = 'embedding:'
        self.embedding_key  = embedding_key
        self.embedding_size = embedding_size

        # tokens
        self.pad_token = self.tokenizer('<pad>', add_special_tokens=False)['input_ids'][0]
        self.end_token = self.tokenizer('</s>' , add_special_tokens=False)['input_ids'][0]
        self.inverse_vocab = { v: k for k, v in self.get_vocab().items() }



    @classmethod
    def from_pretrained(cls,
        tokenizer_dir : Optional[os.PathLike] = None,
        max_length    : int                   = 120,  # alpha=120 | sigma=300 #
        embedding_dir : Optional[os.PathLike] = None,
        embedding_key : str                   = 't5',
        embedding_size: int                   = 4096
    ):
        if tokenizer_dir is None:
            _this_file_dir = os.path.dirname(os.path.realpath(__file__))
            tokenizer_dir = os.path.join(_this_file_dir, 't5tokenizer')

        # TODO: utilizar T5TokenizerFast (?)
        tokenizer = HF_T5Tokenizer.from_pretrained(tokenizer_dir)
        return T5Tokenizer(tokenizer,
                           max_length     = max_length,
                           embedding_dir  = embedding_dir,
                           embedding_size = embedding_size,
                           embedding_key  = embedding_key)



    def get_embeddings(self, embedding_name:str):
        if embedding_name.startswith(self.embedding_tag):
            embedding_name = embedding_name[len(self.embedding_tag):]
        embedding = comfy_load_embed(embedding_name,
                                     self.embedding_dir,
                                     self.embedding_size,
                                     self.embedding_key)
        return embedding



    def tokenize_with_weights(self,
                              text           : Union[str, list],
                              return_word_ids: bool = False
                              ):
        '''
        Convert a text/prompt into a list of (token, weight, word_id) elements.
        The input text can be a string or a list of strings (for batch tokenization).
        In the output (token, weight, word_id):
         - 'token' can be either integer tokens or pre-computed T5 tensors.
         - 'weight' is the user-assigned weight.
         - 'word_id' is an integer indicating the word to which the token belongs,
            where word_id=0 is reserved for non-word tokens.

        The returned list has dimensions of (batch_size, max_length).
        '''
        output_batch = []
        input_batch  = text if isinstance(text,list) else [text]

        for text in input_batch:
            text = comfy_escape_important(text)
            parsed_weights = comfy_token_weights(text, 1.0)

            tokens = []
            for segment, weight in parsed_weights:
                segment = comfy_unescape_important(segment).replace('\n',' ')
                words   = [word for word in segment.split() if word]
                for word_idx, word in enumerate(words):

                    # embeddings
                    if word.startswith(self.embedding_tag):
                        # TODO: procesar embeddings
                        _ = self.get_embeddings(self, word)
                        continue

                    # tokenize word
                    word_tokens = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                    if return_word_ids:
                        tokens.extend([ (token, weight, word_idx+1) for token in word_tokens ])
                    else:
                        tokens.extend([ (token, weight) for token in word_tokens ])

            # expande o limita la cantidad de tokens
            # para sea exactamente = self.max_length-1 + END_TOKEN
            tokens_left = self.max_length - 1 - len(tokens)
            if tokens_left>0:
                tokens.append( (self.end_token, 1.0, 0) )
                tokens.extend( [(self.pad_token, 1.0, 0)] * tokens_left )
            elif tokens_left<0:
                tokens = tokens[:self.max_length-1]
                tokens.append( (self.end_token, 1.0, 0) )

            output_batch.append( tokens )

        return output_batch


    # Returns the vocabulary as a dictionary of token to index.
    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()


    def untokenize(self,
                   token_weight_pairs: List,
                   return_word_weights: bool = False
    ) -> List:
        """
        Converts a list of (token, weight) pairs into a list of ((..data..), word) tuples.

        Args:
            token_weight_pairs (List): A list of (token, weight) pairs, where
                token is an integer representing a token, and weight is a
                number representing the weight associated with that token.
            return_word_weights (bool):
                si True entonces retornara una lista con (word, height) pairs

        Returns:
            List: A list of ((..data..), word) tuples, where word is the string
                representation of the token according to the vocabulary.
        """
        if return_word_weights:
            return list(map( lambda tw: (self.inverse_vocab[tw[0]], tw[1]), token_weight_pairs ))
        else:
            return list(map( lambda tw: (tw, self.inverse_vocab[tw[0]]), token_weight_pairs ))


#============================= T5EncoderModel ==============================#
class T5EncoderModel:


    def __init__(self,
                 t5encoder,
                 max_length = 120, # alpha=120 / sigma=300 !!!
                 num_layers = 24,
                 frozen     = False
                 ):
        self.t5encoder    = t5encoder
        self.max_length   = max_length
        self.num_layers   = num_layers
        self.empty_tokens = [[0] * self.max_length] # <pad> token
        self.empty_vector = None
        if frozen:
            self.freeze()


    @classmethod
    def from_safetensors(self,
                         safetensors_path: Union[os.PathLike, list],
                         model_class     : str  = 'xxl',
                         max_length      : int  = 120,  # alpha=120 / sigma=300 !!!
                         frozen          : bool = True,
                         device          : str  = 'cpu'
                         ):
        model_class = model_class.lower()
        model       = None
        assert model_class == 'xxl', 'xxl es el unico model t5 soportado hasta el momento'

        if not isinstance(safetensors_path, list) and \
           not isinstance(safetensors_path, tuple):
            safetensors_path = [ safetensors_path ]

        config = T5PredefinedConfigs[model_class]
        model_config = HF_T5Config(**config)
        with torch.device('meta'):
            model = HF_T5EncoderModel(model_config)

        for filepath in safetensors_path:
            print(f'Loading {filepath}')
            state_dict = {}
            with safe_open(filepath, framework='pt', device=device) as f:
                for key in f.keys():
                    # print(f"  - loading tensor {key}")
                    state_dict[key] = f.get_tensor(key)

            result = model.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict
            gc.collect()
            if isinstance(result,tuple):
                print("PixArt T5 unexpected keys:", result.unexpected_keys)

        return T5EncoderModel(model, max_length=max_length, num_layers=24, frozen=frozen)


    def encode( self, input_ids ):
        device    = self.t5encoder.get_input_embeddings().weight.device
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.zeros_like(input_ids)
        max_token = 1 # </s> token
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if input_ids[x, y] == max_token:
                    break

        outputs = self.t5encoder(input_ids=input_ids, attention_mask=attention_mask)

        z = outputs['last_hidden_state']
        z.detach().cpu().float()
        return z


    def encode_with_weights( self, batch_of_tokens_with_weights ):

        # separa tokes y pesos
        input_ids     = []
        input_weights = []
        for element in batch_of_tokens_with_weights:
            input_ids.append(     list(map(lambda a: a[0], element)) )
            input_weights.append( list(map(lambda a: a[1], element)) )

        # cachea el vector de prompt vacio
        if self.empty_vector is None:
            self.empty_vector = self.encode( self.empty_tokens ).squeeze(0)
        empty_vector = self.empty_vector

        # generar la representacion vectorial del bach de tokens suministrado
        #   input_ids.shape    = (batch_size, sequence_length)
        #   output_batch.shape = (batch_size, sequence_length, embedding_size)
        output_batch = self.encode( input_ids = input_ids )

        # Aplica los pesos a los context_embeddings
        # ATTENTION: se estan aplicando los pesos a la SALIDA del encoder T5
        #            y utilizando la implementacion de comfy.
        # Nota:
        #  Los embeddings contextualizados son representaciones vectoriales
        #  de palabras que incorporan información semántica y sintáctica del
        #  contexto circundante.
        #  A diferencia de los embeddings tradicionales, su carácter más
        #  conceptual los hace ideales para procesamiento de lenguaje natural.
        #
        # T5 permitiria aplicar los pesos:
        #    - a los embeddings de cada token (antes de ingresar al encoder T5)
        #    * a los contextualized embeddings (luego de salir del encoder T5)
        # Weight interpretation:
        #    * comfy: vectors are lerped between the prompt and an empty prompt
        #    - A1111: vectors are scaled by their weight
        #
        weighted_batch = []
        for i, context_embeddings in enumerate(output_batch):
            #  context_embeddings.shape = (sequence_length=120, embedding_size=4096)
            weights = torch.Tensor( input_weights[i] )
            weighted_embeddings = ( context_embeddings - empty_vector ) * weights.unsqueeze(1) + empty_vector
            weighted_batch.append( weighted_embeddings )

        # si por algun motivo el batch a encodear no tiene ningun elemento
        # retornar el vector de prompt vacio
        if len(weighted_batch) == 0:
            return empty_vector.cpu()

        # convierte la lista a vector
        weighted_batch = torch.stack(weighted_batch, dim=0)

        # TODO: mejorar esto. [[ tal vez usando attention_mask (?) ]]
        keep_index = sum([sum([1 for y in x if y[0] != 0]) for x in batch_of_tokens_with_weights])
        print("## keep_index:", keep_index)
        weighted_batch = weighted_batch[:, :keep_index, :]
        return weighted_batch, "" # first_pooled #


    # Freeze all params for inference.
    def freeze(self) -> None:
        for param in self.t5encoder.parameters():
            param.requires_grad = False
        self.t5encoder.eval()


    # Unfreeze all parameters for training.
    def unfreeze(self) -> None:
        for param in self.t5encoder.parameters():
            param.requires_grad = True
        self.t5encoder.train()


    # Copy parameters and buffers from state_dict into the t5encoder
    #  - state_dict: a dict containing parameters and persistent buffers.
    #  - strict: whether to strictly enforce that the keys in state_dict match the keys in t5encoder
    # Returns: NamedTuple with 'missing_keys' and 'unexpected_keys' fields.
    def load_state_dict(self, state_dict, strict=False):
        return self.t5encoder.load_state_dict(state_dict, strict=strict)


    # Return a dictionary containing references to the whole state of the T5 encoder
    def state_dict(self, *args, **kwargs):
        return self.t5encoder.state_dict(*args, **kwargs)


    # Move and/or cast the parameters and buffers of the T5 encoder
    def to(self, *args, **kwargs):
        self.t5encoder.to(*args, **kwargs)


