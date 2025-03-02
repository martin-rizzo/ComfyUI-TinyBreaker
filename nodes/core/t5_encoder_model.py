
"""
File    : t5_encoder_model.PY
Purpose : Experimental T5 encoder.
    - supports storing model weights on one device and performing inference on another.
    - supports storing model in a dtype (e.g. float8) and perform inference on another (e.g. bfloat16)
    - supports preloading layers in GPU ahead of time during inference.
    - the code has minimal dependencies to be integrated into any project.

Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import time
import math
import torch
from torch    import Tensor
from torch.nn import functional as F

# import traceback
# def _print_traceback(title: str = ""):
#     stack_summary = traceback.extract_stack()
#     print()
#     print(f"{title.upper()} call stack:" if title else "call stack:")
#     for filename, line_number, function_name, text in stack_summary:
#         print(f"  File: {filename}, Line: {line_number}, Function: {function_name}, Code: {text}")
#     print()

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# The data types that are supported by PyTorch for computations.
PYTORCH_COMPUTABLE_DATA_TYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)

# The possible activation functions used in the Transformer model
_ACTIVATIONS_BY_NAME = {
    "linear"           : lambda x: x,
    "gelu"             : F.gelu,
    "gelu_new"         : lambda x: x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))),
    "gelu_fast"        : lambda x: x * 0.5 * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))),
    "gelu_python"      : lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu"       : lambda x: x * torch.sigmoid(1.702 * x),
    "relu"             : F.relu,
    "relu6"            : F.relu6,
    "sigmoid"          : F.sigmoid,
    "silu"             : F.silu,
    "swish"            : F.silu,
    "tanh"             : torch.nn.Tanh,
    "prelu"            : torch.nn.PReLU,
}


class InferenceResourceManager:
    """
    A simple resource manager for efficient memory usage during model inference.

    This class provides a mechanism to optimize the memory usage and execution
    speed during inference by lazily loading and unloading model parameters
    to/from GPU memory. It works in conjunction with model subclasses that
    inherit from both `torch.nn.Module` and `InferenceResourceManager`.
    """
    def __init__(self):
        super().__init__()
        self._infer_weight            : Tensor | None = None
        self._infer_bias              : Tensor | None = None
        self._persistent_infer_tensors: bool          = False

    def prepare_for_inference(self, device: torch.device, dtype: torch.dtype, is_persistent: bool | None = None) -> None:
        """
        Prepares the parameters for inference on a specified device and data type.

        This method can be called much before the inference process to ensure
        that the model parameters are loaded onto the GPU where the inference
        will take place.
        """
        if hasattr(self, 'weight') and self._infer_weight is None:
            self._infer_weight = self.weight.to(device, dtype, non_blocking=True) if self.weight is not None else None
        if hasattr(self, 'bias') and self._infer_bias is None:
            self._infer_bias = self.bias.to(device, dtype, non_blocking=True) if self.bias is not None else None
        if is_persistent is not None:
            self._persistent_infer_tensors = is_persistent

    def prepare_for_inference_with(self, x: Tensor, is_persistent: bool | None = None, /) -> None:
        """Prepares the parameters for inference with a given tensor."""
        self.prepare_for_inference(x.device, x.dtype, is_persistent)

    def begin_inference(self, device: torch.device, dtype: torch.dtype) -> tuple:
        """Begins inference by preparing the parameters for GPU computations."""
        self.prepare_for_inference(device, dtype)
        return self._infer_weight, self._infer_bias

    def end_inference(self):
        """Ends inference by unloading parameters from the GPU memory if they are not persistent."""
        if not self._persistent_infer_tensors:
            self._infer_weight = None
            self._infer_bias   = None


class Custom_nn:

    class Linear(torch.nn.Linear, InferenceResourceManager):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            InferenceResourceManager.__init__(self)

        def forward(self, input: Tensor) -> Tensor:
            weight, bias = self.begin_inference(input.device, input.dtype)
            output = F.linear(input, weight, bias)
            self.end_inference()
            return output

        def reset_parameters(self):
            pass


    class Embedding(torch.nn.Embedding, InferenceResourceManager):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            InferenceResourceManager.__init__(self)

        def forward(self, input: Tensor, dtype: torch.dtype) -> Tensor:
            weight, _ = self.begin_inference(input.device, dtype)
            output = F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            self.end_inference()
            return output

        def reset_parameters(self):
            pass


    class Parameter(torch.nn.Parameter):
        pass

    class Dropout(torch.nn.Dropout):
        pass



#--------------------------------- HELPERS ---------------------------------#

def _get_4d_attention_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """
    Creates a 4D mask of shape `[batch_size, 1, seq_length, seq_length]`
    from a 2D mask of shape `[batch_size, seq_length]`
    """
    if attention_mask.dim() != 2:
        return attention_mask

    assert len(attention_mask.shape) == 2
    batch_size = attention_mask.shape[0]
    seq_length = attention_mask.shape[1]
    min_dtype  = torch.finfo(dtype).min

    mask4d = 1.0 - attention_mask.to(dtype).reshape(batch_size, 1, -1, seq_length).expand(batch_size, 1, seq_length, seq_length)
    mask4d = mask4d.masked_fill(mask4d.to(torch.bool), min_dtype)
    return mask4d


def _relative_position_buckets(query_length: int,
                               key_length  : int,
                               *,
                               num_buckets  : int,
                               max_distance : int,
                               bidirectional: bool,
                               device       : torch.device
                               ) -> Tensor:
    """
    Computes relative position buckets for a given query and key sequence.

    This function determines the relative distances between each pair of query
    and key tokens and assigns these distances to integer buckets. To effectively
    capture both short-range and long-range relationships, it employs a hybrid
    approach: a portion of the distance values are represented linearly, while
    another portion is represented logarithmically.
     - Distances greater than `num_buckets/2` are mapped logarithmically.
     - Distances exceeding `max_distance` are all mapped to the same bucket.

    Args:
        query_length   (int): Length of the query sequence.
        key_length     (int): Length of the key sequence.
        num_buckets    (int): Number of relative position buckets to generate.
        max_distance   (int): Maximum allowed distance between tokens.
        bidirectional (bool): Whether to consider both positive and negative
                              distances (True) or only positive distances (False).
                              Defaults to True.
        device (torch.device): Device to run the computation on (CPU or GPU).

    Returns:
        A 2D tensor of shape [query_length, key_length] representing the bucket
        indices for each query-key pair.
    """
    assert bidirectional == True, "only bidirectional relative position buckets are supported"
    assert num_buckets % 2 == 0, "num_buckets must be even"

    _query_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    _key_position   = torch.arange(key_length  , dtype=torch.long, device=device)[None, :]
    query_to_key_distance = _key_position - _query_position                       # INT64[query_length, key_length]

    relative_buckets = 0

    if bidirectional:
        # if bidirectional, we care about both positive and negative distances
        # but the maximum bucket value is only HALF of the number of buckets provided:
        #  - negative distances will have values of [max_bucket-1,      0        ]
        #  - positive distances will have values of [max_bucket  , 2*max_bucket-1]
        max_bucket = num_buckets // 2
        relative_buckets += (query_to_key_distance > 0).to(torch.long) * max_bucket
    else:
        # if not bidirectional, we only care about distances < 0
        # but the maximum bucket value is the number of buckets provided
        max_bucket = num_buckets
        query_to_key_distance = torch.min(query_to_key_distance, torch.zeros_like(query_to_key_distance))

    # distances must always be positive in the range [0, inf]
    query_to_key_distance = torch.abs(query_to_key_distance)

    # sets the limit between the half that contains linear distance
    # and the half that contains logaritmic distance
    limit = max_bucket // 2

    # calculate the logarithmic distances
    logarithmic_distance = (                                                      # FLOAT[query_length, key_length]
        torch.log(query_to_key_distance.float() / limit) / math.log(max_distance / limit) * (max_bucket - limit) )
    logarithmic_distance = limit + logarithmic_distance.to(torch.long)

    # clamp logarithmic distances to the range [0, max_bucket - 1]
    logarithmic_distance = torch.min(                                             # INT64[query_length, key_length]
        logarithmic_distance, torch.full_like(logarithmic_distance, max_bucket - 1) )

    # compose linear and logarithmic distances depending if distance exceeds limit or not
    relative_buckets += torch.where( query_to_key_distance < limit, query_to_key_distance, logarithmic_distance )
    return relative_buckets


class _RMSNorm(torch.nn.Module):
    """
    A RMS-based normalization layer, devoid of learned bias and mean subtraction

    This module normalizes the input tensor by scaling each feature with its
    root mean square value, calculated across the last dimension (typically
    representing samples).
    It is implemented as described in the paper "RMS Layer Normalization":
     - https://arxiv.org/pdf/1910.07467.pdf

    Args:
        dim    : The number of features in both the input and output tensors.
        epsilon: A small value added to the variance to prevent division by zero.
        dtype  : The data type of the weight tensor. Defaults to torch.float32.
        nn (optional): The neural network implementation to use.
    """
    def __init__(self,
                 dim: int,
                 /,*,
                 epsilon: float,
                 dtype: torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        self.weight           = nn.Parameter( torch.empty(dim, dtype=dtype) )
        self.variance_epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(x) * x

    def prepare_for_inference_with(self, x: Tensor, is_persistent: bool | None = None, /) -> None:
        # ??
        pass


#=========================== FEED FORWARD LAYER ============================#

class _DenseActDense(torch.nn.Module):

    def __init__(self,
                 dim          : int,
                 hidden_dim   : int,
                 activation_fn: str,
                 dropout_rate : float,
                 /,*,
                 dtype: torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        self.wi      = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.act     = _ACTIVATIONS_BY_NAME[activation_fn]
        self.dropout = nn.Dropout(dropout_rate)
        self.wo      = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)

    def forward(self, x):
        x = self.wi(x)
        x = self.act(x)
        #x= self.dropout(x)
        x = self.wo(x)
        return x

    def prepare_for_inference_with(self, *args):
        self.wi.prepare_for_inference_with(*args)
        self.wo.prepare_for_inference_with(*args)


class _DenseGatedActDense(torch.nn.Module):

    def __init__(self,
                 dim          : int,
                 hidden_dim   : int,
                 activation_fn: str,
                 dropout_rate : float,
                 /,*,
                 dtype: torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        self.wi_1    = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.wi_0    = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.act     = _ACTIVATIONS_BY_NAME[activation_fn]
        self.dropout = nn.Dropout(dropout_rate)
        self.wo      = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.wi_1(x)
        x = self.wi_0(x)
        x = self.act(x)
        x *= gate
        #x= self.dropout(x)
        x = self.wo(x)
        return x

    def prepare_for_inference_with(self, *args):
        self.wi_1.prepare_for_inference_with(*args)
        self.wi_0.prepare_for_inference_with(*args)
        self.wo.prepare_for_inference_with(*args)


#--------------------------------------
class FeedForwardLayer(torch.nn.Module):

    def __init__(self,
                 *,
                 model_dim         : int,
                 ff_dim            : int,
                 ff_activation_fn  : str,
                 ff_is_gated       : bool,
                 layer_norm_epsilon: float,
                 dropout_rate      : float,
                 dtype             : torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        DenseActDense = _DenseGatedActDense if ff_is_gated else _DenseActDense

        self.layer_norm     = _RMSNorm(model_dim, epsilon=layer_norm_epsilon, dtype=dtype, nn=nn)
        self.DenseReluDense = DenseActDense(model_dim, ff_dim, ff_activation_fn, dropout_rate, dtype=dtype, nn=nn)
        self.dropout        = nn.Dropout(dropout_rate)

    def forward(self, x):
        # REF:
        #  x      -> [batch_size, seq_length, model_dim]
        #  output -> [batch_size, seq_length, model_dim]
        residual = x
        x = self.layer_norm(x)
        x = self.DenseReluDense(x)
        #x= self.dropout(x)
        x += residual
        return x

    def prepare_for_inference_with(self, *args):
        self.layer_norm.prepare_for_inference_with(*args)
        self.DenseReluDense.prepare_for_inference_with(*args)


#========================== SELF ATTENTION LAYER ===========================#

class _SelfAttention(torch.nn.Module):

    def __init__(self,
                 model_dim: int,
                 inner_dim: int,
                 num_heads: int,
                 /,*,
                 has_relative_attention_bias: bool,
                 relative_attn_num_buckets  : int,
                 relative_attn_max_distance : int,
                 dtype: torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        assert (inner_dim % num_heads) == 0, f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads               = num_heads
        self.inner_dim               = inner_dim
        self.relative_attention_bias = None

        # relative attention bias
        if has_relative_attention_bias:
            self.relative_attention_bias    = nn.Embedding(relative_attn_num_buckets, num_heads, dtype=dtype)
            self.relative_attn_num_buckets  = relative_attn_num_buckets
            self.relative_attn_max_distance = relative_attn_max_distance

        # attention qkv projection
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype)

    def forward(self,
                x            : Tensor,
                mask         : Tensor = None,
                position_bias: Tensor = None,
                ) -> Tensor:
        # REF:
        #  x         -> [batch_size, seq_length, model_dim]
        #  output    -> [batch_size, seq_length, model_dim]
        #  past_bias -> [1, num_heads, seq_length, seq_length]
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        num_heads  = self.num_heads
        inner_dim  = self.inner_dim
        head_dim   = inner_dim // self.num_heads

        # if this layer has a `relative_attention_bias` then the position bias is calculated,
        # otherwise the provided one is used (probably a position bias calculated in previous layer)
        if self.relative_attention_bias is not None:
            _position_buckets = _relative_position_buckets(seq_length, seq_length,    # INT64[seq_length, seq_length]
                                                           num_buckets   = self.relative_attn_num_buckets,
                                                           max_distance  = self.relative_attn_max_distance,
                                                           bidirectional = True,
                                                           device = x.device
                                                           )
            position_bias = self.relative_attention_bias(_position_buckets,      # [seq_length, seq_length, num_heads]
                                                         dtype = x.dtype
                                                         )
            position_bias = position_bias.permute(2, 0, 1).unsqueeze(0)          # [1, num_heads, seq_length, seq_length]

        # attention mask is `position_bias` or `mask`` (the one that
        # has been supplied), and if both are present they are added
        if mask is None:
            attn_mask = position_bias
        elif position_bias is None:
            attn_mask = mask
        else:
            attn_mask = mask + position_bias

        # project query, key and value
        q: Tensor = self.q(x)                                                    # [batch_size, seq_length, inner_dim]
        k: Tensor = self.k(x)                                                    # [batch_size, seq_length, inner_dim]
        v: Tensor = self.v(x)                                                    # [batch_size, seq_length, inner_dim]

        # prepare for the multi-head dot product attention
        q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        v = v.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]

        # do the dot product attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        x = x.transpose(1, 2).reshape(batch_size, seq_length, inner_dim)         # [batch_size, seq_length, inner_dim]
        x = self.o(x)                                                            # [batch_size, seq_length, model_dim]
        return x, position_bias

    def prepare_for_inference_with(self, *args):
        if self.relative_attention_bias:
            self.relative_attention_bias.prepare_for_inference_with(*args)
        self.q.prepare_for_inference_with(*args)
        self.k.prepare_for_inference_with(*args)
        self.v.prepare_for_inference_with(*args)
        self.o.prepare_for_inference_with(*args)


#----------------------------------------
class SelfAttentionLayer(torch.nn.Module):
    """
    The self-attention mechanism used by the T5 encoder.

    This module provides a flexible, parameterized structure for self-attention
    operations. It leverages multiple attention heads and relative positional
    encodings to capture complex dependencies in sequential data.

    Args:
        model_dim                    (int): The dimensionality of the embeddings.
        inner_dim                    (int): The hidden dimension for the self attention mechanism.
        num_heads                    (int): Number of attention heads in multi-head attention mechanisms.
        layer_norm_epsilon         (float): A small value added in the normalization layer to prevent division by zero.
        has_relative_attention_bias (bool): If True, this block will have its own relative attention bias.
                                            If False, it shares the same relative bias with other blocks.
        relative_attn_num_buckets    (int): Number of buckets for relative positional encoding.
        relative_attn_max_distance   (int): Maximum distance in the relative positional encoding.
        dropout_rate               (float): Dropout rate applied after each layer.
        dtype                (torch.dtype): The data type used for storing model tensors.
        nn                      (optional): The neural network library to employ (e.g., `torch.nn`).
    """
    def __init__(self,
                 *,
                 model_dim                  : int,
                 inner_dim                  : int,
                 num_heads                  : int,
                 layer_norm_epsilon         : float,
                 has_relative_attention_bias: bool,
                 relative_attn_num_buckets  : int,
                 relative_attn_max_distance : int,
                 dropout_rate               : float,
                 dtype                      : torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        self.SelfAttention = _SelfAttention(model_dim, inner_dim, num_heads,
                                            has_relative_attention_bias = has_relative_attention_bias,
                                            relative_attn_num_buckets   = relative_attn_num_buckets,
                                            relative_attn_max_distance  = relative_attn_max_distance,
                                            dtype                       = dtype,
                                            nn=nn
                                            )
        self.layer_norm    = _RMSNorm(model_dim, epsilon=layer_norm_epsilon, dtype=dtype, nn=nn)
        self.dropout       = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, position_bias=None):
        # REF:
        # x         -> [batch_size, seq_length, model_dim]
        # past_bias -> [batch_size, num_heads, seq_length, seq_length]
        residual = x
        x = self.layer_norm(x)
        x, position_bias = self.SelfAttention(x, mask, position_bias)
        #x= self.dropout(x)
        x += residual
        return x, position_bias

    def prepare_for_inference_with(self, *args):
        self.layer_norm.prepare_for_inference_with(*args)
        self.SelfAttention.prepare_for_inference_with(*args)


#=========================== T5 STACK OF BLOCKS ============================#

class _T5Block(torch.nn.Module):
    def __init__(self,
                 *,
                 model_dim                  : int,
                 inner_dim                  : int,
                 ff_dim                     : int,
                 ff_activation_fn           : str,
                 ff_is_gated                : bool,
                 num_heads                  : int,
                 layer_norm_epsilon         : float,
                 has_relative_attention_bias: bool,
                 relative_attn_num_buckets  : int,
                 relative_attn_max_distance : int,
                 dropout_rate               : float,
                 dtype: torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn
        self.layer = torch.nn.ModuleList()

        self.layer.append( SelfAttentionLayer(
                                    model_dim                   = model_dim,
                                    inner_dim                   = inner_dim,
                                    num_heads                   = num_heads,
                                    layer_norm_epsilon          = layer_norm_epsilon,
                                    has_relative_attention_bias = has_relative_attention_bias,
                                    relative_attn_num_buckets   = relative_attn_num_buckets,
                                    relative_attn_max_distance  = relative_attn_max_distance,
                                    dropout_rate                = dropout_rate,
                                    dtype                       = dtype,
                                    nn=nn
                                    ))
        self.layer.append( FeedForwardLayer(
                                    model_dim          = model_dim,
                                    ff_dim             = ff_dim,
                                    ff_activation_fn   = ff_activation_fn,
                                    ff_is_gated        = ff_is_gated,
                                    layer_norm_epsilon = layer_norm_epsilon,
                                    dropout_rate       = dropout_rate,
                                    dtype              = dtype,
                                    nn=nn
                                    ))

    def forward(self, x, mask=None, position_bias=None):
        x, position_bias = self.layer[ 0](x, mask, position_bias)  # self attention
        x                = self.layer[-1](x)                       # feed forward
        return x, position_bias

    def prepare_for_inference_with(self, *args):
        for layer in self.layer:
            layer.prepare_for_inference_with(*args)


#-------------------------------------
class T5StackOfBlocks(torch.nn.Module):
    """
    A stack of transformer blocks within the T5 encoder architecture.

    Each block in this stack implements self-attention and feed-forward networks,
    forming a deep structure that can handle long-range dependencies in text data
    effectively.

    Args:
        model_dim        (int): The dimensionality of the embeddings.
        inner_dim        (int): The hidden dimension for the self attention mechanism.
        num_heads        (int): Number of attention heads in multi-head attention mechanisms.
        num_layers       (int): Total number of blocks in the stack.
        ff_dim           (int): Dimension of the feed-forward network layer.
        ff_is_gated     (bool): Whether to use a gated linear unit in the feed-forward layer.
        ff_activation_fn (str): Activation function used in feed-forward networks.
        layer_norm_epsilon           (float): Epsilon value for Layer Scaling in Layer Normalization.
        all_layers_have_relative_bias (bool): If True, each block will have its own relative attention bias;
                                              If False, the same relative bias is applied across all blocks.
        relative_attn_num_buckets      (int): Number of buckets for relative positional encoding.
        relative_attm_max_distance     (int): Maximum distance in the relative positional encoding.
        dropout_rate   (float): Dropout rate applied after each layer.
        dtype    (torch.dtype): The data type used for storing model tensors.
        nn          (optional): The neural network library to employ (e.g., `torch.nn`).
    """
    def __init__(self,
                 *,
                 model_dim                    : int,
                 inner_dim                    : int,
                 num_heads                    : int,
                 num_layers                   : int,
                 ff_dim                       : int,
                 ff_is_gated                  : bool,
                 ff_activation_fn             : str,
                 layer_norm_epsilon           : float,
                 all_layers_have_relative_bias: bool,
                 relative_attn_num_buckets    : int,
                 relative_attm_max_distance   : int,
                 dropout_rate                 : float,
                 dtype                        : torch.dtype,
                 nn
                 ):
        super().__init__()
        if nn is None:
            nn = Custom_nn

        self.block = torch.nn.ModuleList([
            _T5Block(model_dim                  = model_dim,
                    inner_dim                   = inner_dim,
                    ff_dim                      = ff_dim,
                    ff_activation_fn            = ff_activation_fn,
                    ff_is_gated                 = ff_is_gated,
                    num_heads                   = num_heads,
                    layer_norm_epsilon          = layer_norm_epsilon,
                    has_relative_attention_bias = (layer == 0) or all_layers_have_relative_bias,
                    relative_attn_num_buckets   = relative_attn_num_buckets,
                    relative_attn_max_distance  = relative_attm_max_distance,
                    dropout_rate                = dropout_rate,
                    dtype                       = dtype,
                    nn=nn
                    )
             for layer in range(num_layers)
        ])
        self.final_layer_norm = _RMSNorm(model_dim, epsilon=layer_norm_epsilon, dtype=dtype, nn=nn)
        self.dropout          = nn.Dropout(dropout_rate)


    def forward(self,
                x                              : torch.Tensor,
                attention_mask                 : torch.Tensor = None,
                return_intermediate            : bool         = False,
                intermediate_index             : int          = -2,
                intermediate_must_be_normalized: bool         = False,
                precharge_depth                : int          = 2,
                ) -> Tensor | tuple[Tensor]:

        # the `intermediate_index` parameter can have a negative value,
        # the same logic as Python indexing is used
        if intermediate_index < 0:
            intermediate_index += len(self.block)

        # adjust the `attention_mask` parameter to always be 4D
        if attention_mask:
            attention_mask = _get_4d_attention_mask(attention_mask, x.dtype)

        # prepare the first `precharge_depth` blocks for inference
        for index in range(precharge_depth):
            self.block[index].prepare_for_inference_with(x, True)

        # iterates over each transformer block in the stack
        intermediate  = None
        position_bias = None
        last_index    = len(self.block)-1
        for index, block in enumerate(self.block):

            # prepare a future block for inference
            if (index+precharge_depth) <= last_index:
                empty_cache()
                self.block[index+precharge_depth].prepare_for_inference_with(x)

            # apply attention and feed-forward on the input
            x, position_bias = block(x, attention_mask, position_bias)
            if return_intermediate and (index == intermediate_index):
                intermediate = x.clone()

        # apply the final normalization layer
        x = self.final_layer_norm(x)
        if (intermediate is not None) and intermediate_must_be_normalized:
            intermediate = self.final_layer_norm(intermediate)

        if return_intermediate:
            return x, intermediate
        return x


#===========================================================================#
#//////////////////////////// T5 ENCODER MODEL /////////////////////////////#
#===========================================================================#

class T5EncoderModel(torch.nn.Module):

    def __init__(self,
                 *,
                 d_ff                           : int   = 10240,
                 d_kv                           : int   =    64,
                 d_model                        : int   =  4096,
                 dense_act_fn                   : str   = "gelu_pytorch_tanh",
                 layer_norm_epsilon             : float =  1e-6,
                 model_type                     : str   =  "t5",
                 is_gated_act                   : bool  =  True,
                 num_heads                      : int   =    64,
                 num_layers                     : int   =    24,
                 vocab_size                     : int   = 32128,
                 relative_attention_num_buckets : int   =    32,
                 relative_attention_max_distance: int   =   128,
                 dropout_rate                   : float =   0.1,
                 device: str | torch.device = "cpu",
                 dtype :       torch.dtype  = torch.float16,
                 nn = None,
                 **kwargs
                 ):
        """
        Args:
            device       : The device on which to store model parameters.
            dtype        : The data type of model parameters.
            nn (optional): The neural network module implementation to use.
                           Defaults to a custom implementation of `torch.nn`.
                           This parameter allows for injecting custom or optimized
                           implementations of neural network modules (`nn`).
        """
        super().__init__()
        if nn is None:
            nn = Custom_nn
        if isinstance(device, str):
            device = torch.device(device)

        # backward compatibility with older config versions,
        # where the activation function was called "feed_forward_proj"
        dense_act_fn = kwargs.get("feed_forward_proj", dense_act_fn)
        if dense_act_fn == "gated-gelu":
            dense_act_fn = "gelu_new"

        with torch.device(device):
            self.shared  = nn.Embedding(
                                    num_embeddings = vocab_size,
                                    embedding_dim  = d_model,
                                    dtype          = dtype
                                    )
            self.encoder = T5StackOfBlocks(
                                    model_dim                     = d_model,
                                    inner_dim                     = d_kv * num_heads,
                                    num_heads                     = num_heads,
                                    num_layers                    = num_layers,
                                    ff_dim                        = d_ff,
                                    ff_is_gated                   = is_gated_act,
                                    ff_activation_fn              = dense_act_fn,
                                    layer_norm_epsilon            = layer_norm_epsilon,
                                    all_layers_have_relative_bias = (model_type == "umt5"),
                                    relative_attn_num_buckets     = relative_attention_num_buckets,
                                    relative_attm_max_distance    = relative_attention_max_distance,
                                    dropout_rate                  = dropout_rate,
                                    dtype                         = dtype,
                                    nn=nn
                                    )

    def forward(self,
                input_ids,
                *,
                attention_mask                 : torch.Tensor = None,
                return_intermediate            : bool         = False,
                intermediate_index             : int          = -2,
                intermediate_must_be_normalized: bool         = False,
                device                         : torch.device = None,
                dtype                          : torch.dtype  = None,
                ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Forward pass of the T5 encoder.
        Args:
            input_ids                      : the input token ids to encode.
            attention_mask                 : the attention mask to use for masking input tokens.
            return_intermediate            : whether to return intermediate hidden states.
            intermediate_index             : index of intermediate hidden state to return.
            intermediate_must_be_normalized: whether to normalize intermediate hidden states with the final layer norm.
            device                         : the device on which to perform the inference.
            dtype                          : the data type to use for performing the inference (bfloat16 or float32).
        """
        # REF:
        #  input_ids     -> [batch_size, seq_length]
        #  inputs_embeds -> [batch_size, seq_length, model_dim]
        #  return        -> [batch_size, seq_length, model_dim]
        print()
        print("##>> input_ids.shape:", input_ids.shape)
        print("##>> attention_mask.shape:", attention_mask.shape if attention_mask else "None")
        print()
        start_time    = time.time()

        layer_0_dtype    = self.encoder.block[0].layer[0].SelfAttention.k.weight.dtype
        inference_dtype  = layer_0_dtype if layer_0_dtype in PYTORCH_COMPUTABLE_DATA_TYPES else torch.float32
        inference_device = input_ids.device

        # if a device or a data type is specified, use them instead of the default ones
        if device:
            inference_device = torch.device(device) if isinstance(device, str) else device
        if dtype:
            inference_dtype  = dtype

        # get the embedding vectors
        self.shared.prepare_for_inference(inference_device, inference_dtype)
        input_embeds = self.shared(input_ids.to(inference_device), dtype=inference_dtype)

        # hack to try to avoid `nan` values in float8
        if layer_0_dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            input_embeds = torch.nan_to_num(input_embeds)

        # forward pass the embedding vectors through the encoder
        tensor_or_tuple = self.encoder(input_embeds,
                                       attention_mask                  = attention_mask,
                                       return_intermediate             = return_intermediate,
                                       intermediate_index              = intermediate_index,
                                       intermediate_must_be_normalized = intermediate_must_be_normalized,
                                       )

        print(f"##>>>>>>  Execution Time: {time.time() - start_time:.4f} seconds")
        return tensor_or_tuple


    def get_input_embeddings(self):
        return self.shared


    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

