import copy
from typing import Optional, Tuple, Union, Dict
import copy
import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.marian.configuration_marian import MarianConfig
from transformers.models.marian.modeling_marian import (
    MarianMTModel,
    MarianModel,
    shift_tokens_right,
    MarianEncoderLayer,
    MarianDecoderLayer,
    MarianEncoder,
    MarianSinusoidalPositionalEmbedding,
    MarianDecoder,
    MarianPreTrainedModel,
)
from transformers.utils import (
    logging,
)

# def tokenize(tokenizer, device, src_lang, tgt_lang, text, is_target=False, return_offsets_mapping=False):
#     tokenizer.src_lang = tgt_lang if is_target else src_lang
#     return tokenizer(text, return_tensors='pt', return_offsets_mapping=return_offsets_mapping).to(device)
from modeling.modeling_outputs import AdjustableBaseModelOutput, AdjustableBaseModelOutputWithPastAndCrossAttentions, \
    AdjustableSeq2SeqModelOutput, AdjustableSeq2SeqLMOutput

logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class AdjustableMarianAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.disabled_heads = None
        self.disabled_heads_tokens = None
        # not used in the paper
        # self.quantized_heads = None
        # self.quantize_all_heads = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            head_disturbance_mask: Optional[torch.Tensor] = None,
            head_disturbance_value: float = 0.5,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.disabled_heads is not None:
            if self.disabled_heads_tokens is not None:
                heads_mask = torch.ones((1, self.num_heads, tgt_len, 1),
                                        dtype=attn_weights.dtype, device=attn_weights.device)
                heads_mask[0, self.disabled_heads, self.disabled_heads_tokens, :] = 0
            else:
                heads_mask = torch.ones((1, self.num_heads, 1, 1), dtype=attn_weights.dtype, device=attn_weights.device)
                heads_mask[0, self.disabled_heads, ...] = 0

            attn_weights = heads_mask * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if head_disturbance_mask is not None:
            if head_disturbance_mask.size() != (bsz, self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Head disturbance mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {head_disturbance_mask.size()}"
                )
            rev_mask = 1 - head_disturbance_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            n = torch.maximum(torch.sum(head_disturbance_mask, dim=-1),
                              torch.tensor(1.0, device=head_disturbance_mask.device))
            b = head_disturbance_value
            a = torch.sum((torch.exp(attn_weights) * rev_mask), dim=-1)
            y = a * b / ((1 - b) * n)
            x = torch.unsqueeze(torch.log(y), dim=-1)
            attn_weights = attn_weights * rev_mask + head_disturbance_mask * x
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        pre_attn_weights = attn_weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # if self.quantized_heads is not None:
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     head_weights = attn_weights[:, self.quantized_heads]
        #     head_maxes = torch.argmax(head_weights, dim=-1)
        #     head_quantized = torch.nn.functional.one_hot(head_maxes, head_weights.shape[-1]).to(head_weights.dtype)
        #     attn_weights[:, self.quantized_heads, ...] = head_quantized
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # elif self.quantize_all_heads:
        #     # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_maxes = torch.argmax(attn_weights, dim=-1)
        #     attn_weights = torch.nn.functional.one_hot(attn_maxes, attn_weights.shape[-1]).to(attn_weights.dtype)
        #     # attn_weights[:, self.quantized_heads, ...] = head_quantized
        #     # attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * layer_head_mask.view(bsz, self.num_heads, tgt_len,
                                                                                    src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
            pre_attn_weights = pre_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
            pre_attn_weights = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value, pre_attn_weights


class AdjustableMarianEncoderLayer(MarianEncoderLayer):
    def __init__(self, config: MarianConfig):
        nn.Module.__init__(self)
        self.embed_dim = config.d_model
        self.self_attn = AdjustableMarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            layer_head_mask: torch.FloatTensor,
            head_disturbance_mask: Optional[torch.Tensor] = None,
            head_disturbance_value: float = 0.5,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _, pre_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            head_disturbance_mask=head_disturbance_mask,
            head_disturbance_value=head_disturbance_value,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights, pre_attn_weights)

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->Marian
class AdjustableMarianDecoderLayer(MarianDecoderLayer):
    def __init__(self, config: MarianConfig):
        nn.Module.__init__(self)
        self.embed_dim = config.d_model

        self.self_attn = AdjustableMarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = AdjustableMarianAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            self_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            self_attention_head_disturbance_value: float = 0.5,
            cross_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            cross_attention_head_disturbance_value: float = 0.5,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor,
               Optional[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value, self_pre_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            head_disturbance_mask=self_attention_head_disturbance_mask,
            head_disturbance_value=self_attention_head_disturbance_value,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        cross_pre_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value, cross_pre_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                head_disturbance_mask=cross_attention_head_disturbance_mask,
                head_disturbance_value=cross_attention_head_disturbance_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, self_pre_attn_weights, cross_pre_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class AdjustableMarianEncoder(MarianEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MarianEncoderLayer`].

    Args:
        config: MarianConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        MarianPreTrainedModel.__init__(self, config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList([AdjustableMarianEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     return self.embed_tokens
    #
    # def set_input_embeddings(self, value):
    #     self.embed_tokens = value
    #
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            head_disturbance_mask: Optional[torch.Tensor] = None,
            head_disturbance_value: float = 0.5,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_pre_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        if head_disturbance_mask is not None:
            assert head_disturbance_mask.size()[0] == (
                len(self.layers)
            ), f"The head_disturbance_mask should be specified for {len(self.layers)} layers, " \
               f"but it is for {head_disturbance_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    # hidden_states: torch.FloatTensor,
                    # attention_mask: torch.FloatTensor,
                    # layer_head_mask: torch.FloatTensor,
                    # head_disturbance_mask: Optional[torch.Tensor] = None,
                    # head_disturbance_value: float = 0.5,
                    # output_attentions: Optional[bool] = False,
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        (head_disturbance_mask[idx] if head_disturbance_mask is not None else None),
                        head_disturbance_value,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        head_disturbance_mask=(head_disturbance_mask[idx]
                                               if head_disturbance_mask is not None else None),
                        head_disturbance_value=head_disturbance_value,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_pre_attentions = all_pre_attentions + (layer_outputs[2],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v
                         for v in [hidden_states, encoder_states, all_attentions, all_pre_attentions]
                         if v is not None)
        return AdjustableBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            pre_attentions=all_pre_attentions
        )


class AdjustableMarianDecoder(MarianDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MarianDecoderLayer`]

    Args:
        config: MarianConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        MarianPreTrainedModel.__init__(self, config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.decoder_vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx
        )
        self.layers = nn.ModuleList([AdjustableMarianDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     return self.embed_tokens
    #
    # def set_input_embeddings(self, value):
    #     self.embed_tokens = value
    #
    # # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    #     # create causal mask
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     combined_attention_mask = None
    #     if input_shape[-1] > 1:
    #         combined_attention_mask = _make_causal_mask(
    #             input_shape,
    #             inputs_embeds.dtype,
    #             device=inputs_embeds.device,
    #             past_key_values_length=past_key_values_length,
    #         )
    #
    #     if attention_mask is not None:
    #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
    #             inputs_embeds.device
    #         )
    #         combined_attention_mask = (
    #             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    #         )
    #
    #     return combined_attention_mask
    #
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            self_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            self_attention_head_disturbance_value: float = 0.5,
            cross_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            cross_attention_head_disturbance_value: float = 0.5,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], AdjustableBaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_self_pre_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        all_cross_pre_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask,
                                         cross_attn_head_mask,
                                         self_attention_head_disturbance_mask,
                                         cross_attention_head_disturbance_mask],
                                        ["head_mask",
                                         "cross_attn_head_mask",
                                         "self_attention_head_disturbance_mask",
                                         "cross_attention_head_disturbance_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                # hidden_states: torch.Tensor,
                # attention_mask: Optional[torch.Tensor] = None,
                # encoder_hidden_states: Optional[torch.Tensor] = None,
                # encoder_attention_mask: Optional[torch.Tensor] = None,
                # layer_head_mask: Optional[torch.Tensor] = None,
                # cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
                # self_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
                # self_attention_head_disturbance_value: float = 0.5,
                # cross_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
                # cross_attention_head_disturbance_value: float = 0.5,
                # past_key_value: Optional[Tuple[torch.Tensor]] = None,
                # output_attentions: Optional[bool] = False,
                # use_cache: Optional[bool] = True,
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    (self_attention_head_disturbance_mask[idx]
                     if self_attention_head_disturbance_mask is not None else None),
                    self_attention_head_disturbance_value,
                    (cross_attention_head_disturbance_mask[idx]
                     if cross_attention_head_disturbance_mask is not None else None),
                    cross_attention_head_disturbance_value,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    self_attention_head_disturbance_mask=(
                        self_attention_head_disturbance_mask[idx]
                        if self_attention_head_disturbance_mask is not None else None
                    ),
                    self_attention_head_disturbance_value=self_attention_head_disturbance_value,
                    cross_attention_head_disturbance_mask=(
                        cross_attention_head_disturbance_mask[idx]
                        if cross_attention_head_disturbance_mask is not None else None
                    ),
                    cross_attention_head_disturbance_value=cross_attention_head_disturbance_value,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_self_pre_attns += (layer_outputs[3],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
                    all_cross_pre_attentions += (layer_outputs[4],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                    all_self_pre_attns,
                    all_cross_pre_attentions,
                ]
                if v is not None
            )
        return AdjustableBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            pre_attentions=all_self_pre_attns,
            cross_pre_attentions=all_cross_pre_attentions,
        )


class AdjustableMarianModel(MarianModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MarianConfig):
        MarianPreTrainedModel.__init__(self, config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        # We always use self.shared for token embeddings to ensure compatibility with all marian models
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            # Since the embeddings are not shared, deepcopy the embeddings here for encoder
            # and decoder to make sure they are not tied.
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None

        self.encoder = AdjustableMarianEncoder(config, encoder_embed_tokens)
        self.decoder = AdjustableMarianDecoder(config, decoder_embed_tokens)

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     # This will return shared embeddings if they are shared else specific to encoder.
    #     return self.get_encoder().get_input_embeddings()
    #
    # def set_input_embeddings(self, value):
    #     if self.config.share_encoder_decoder_embeddings:
    #         self.shared = value
    #         self.encoder.embed_tokens = self.shared
    #         self.decoder.embed_tokens = self.shared
    #     else:  # if not shared only set encoder embeedings
    #         self.encoder.embed_tokens = value
    #
    # def get_decoder_input_embeddings(self):
    #     if self.config.share_encoder_decoder_embeddings:
    #         raise ValueError(
    #             "`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
    #             "is `True`. Please use `get_input_embeddings` instead."
    #         )
    #     return self.get_decoder().get_input_embeddings()
    #
    # def set_decoder_input_embeddings(self, value):
    #     if self.config.share_encoder_decoder_embeddings:
    #         raise ValueError(
    #             "`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings "
    #             "are shared with the encoder. In order to set the decoder input embeddings, you should simply set "
    #             "the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings."
    #         )
    #     self.decoder.embed_tokens = value
    #
    # def get_encoder(self):
    #     return self.encoder
    #
    # def get_decoder(self):
    #     return self.decoder
    #
    # def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
    #     if self.config.share_encoder_decoder_embeddings:
    #         raise ValueError(
    #             "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
    #             "is `True`. Please use `resize_token_embeddings` instead."
    #         )
    #
    #     old_embeddings = self.get_decoder_input_embeddings()
    #     new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    #     self.set_decoder_input_embeddings(new_embeddings)
    #
    #     model_embeds = self.get_decoder_input_embeddings()
    #
    #     if new_num_tokens is None:
    #         return model_embeds
    #
    #     # Update base model and current model config
    #     self.config.decoder_vocab_size = new_num_tokens
    #
    #     # Tie weights again if needed
    #     self.tie_weights()
    #
    #     return model_embeds
    #
    # @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_head_disturbance_mask: Optional[torch.Tensor] = None,
            encoder_head_disturbance_value: float = 0.5,
            decoder_self_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            decoder_self_attention_head_disturbance_value: float = 0.5,
            decoder_cross_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            decoder_cross_attention_head_disturbance_value: float = 0.5,
            encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> AdjustableSeq2SeqModelOutput:
        # r"""
        # Returns:
        #
        # Example:
        #
        # ```python
        # >>> from transformers import AutoTokenizer, MarianModel
        #
        # >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        # >>> model = MarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        #
        # >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        # >>> decoder_inputs = tokenizer(
        # ...     "<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
        # ...     return_tensors="pt",
        # ...     add_special_tokens=False,
        # ... )
        # >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)
        #
        # >>> last_hidden_states = outputs.last_hidden_state
        # >>> list(last_hidden_states.shape)
        # [1, 26, 512]
        # ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                head_disturbance_mask=encoder_head_disturbance_mask,
                head_disturbance_value=encoder_head_disturbance_value,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, AdjustableBaseModelOutput):
            encoder_outputs = AdjustableBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                pre_attentions=encoder_outputs[3] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            self_attention_head_disturbance_mask=decoder_self_attention_head_disturbance_mask,
            self_attention_head_disturbance_value=decoder_self_attention_head_disturbance_value,
            cross_attention_head_disturbance_mask=decoder_cross_attention_head_disturbance_mask,
            cross_attention_head_disturbance_value=decoder_cross_attention_head_disturbance_value,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return AdjustableSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_pre_attentions=decoder_outputs.pre_attentions,
            cross_pre_attentions=decoder_outputs.cross_pre_attentions,
            encoder_pre_attentions=encoder_outputs.pre_attentions,
        )


class AdjustableMarianMTModel(MarianMTModel):
    # base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [
    #     r"final_logits_bias",
    #     r"encoder.version",
    #     r"decoder.version",
    #     r"lm_head.weight",
    #     r"embed_positions",
    #     "encoder.embed_tokens.weight",
    #     "decoder.embed_tokens.weight",
    # ]
    #
    # _keys_to_ignore_on_save = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]

    def __init__(self, config: MarianConfig):
        super(MarianPreTrainedModel, self).__init__(config)
        self.model = AdjustableMarianModel(config)

        target_vocab_size = config.vocab_size if config.share_encoder_decoder_embeddings else config.decoder_vocab_size
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # def get_encoder(self):
    #     return self.model.get_encoder()
    #
    # def get_decoder(self):
    #     return self.model.get_decoder()
    #
    # def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
    #     new_embeddings = super().resize_token_embeddings(new_num_tokens)
    #     if self.config.share_encoder_decoder_embeddings:
    #         self._resize_final_logits_bias(new_num_tokens)
    #     return new_embeddings
    #
    # def _resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
    #     old_embeddings = self.get_input_embeddings()
    #     new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    #     self.set_input_embeddings(new_embeddings)
    #
    #     # update config.decoder_vocab_size if embeddings are tied
    #     if self.config.share_encoder_decoder_embeddings:
    #         self.config.decoder_vocab_size = new_num_tokens
    #
    #     # if word embeddings are not tied, make sure that lm head is resized as well
    #     if (
    #         self.config.share_encoder_decoder_embeddings
    #         and self.get_output_embeddings() is not None
    #         and not self.config.tie_word_embeddings
    #     ):
    #         old_lm_head = self.get_output_embeddings()
    #         new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
    #         self.set_output_embeddings(new_lm_head)
    #
    #     return self.get_input_embeddings()
    #
    # def resize_decoder_token_embeddings(self, new_num_tokens):
    #     if self.config.share_encoder_decoder_embeddings:
    #         raise ValueError(
    #             "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
    #             "is `True`. Please use `resize_token_embeddings` instead."
    #         )
    #
    #     old_embeddings = self.model.get_decoder_input_embeddings()
    #     new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    #     self.model.set_decoder_input_embeddings(new_embeddings)
    #
    #     # if word embeddings are not tied, make sure that lm head is resized as well
    #     if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
    #         old_lm_head = self.get_output_embeddings()
    #         new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
    #         self.set_output_embeddings(new_lm_head)
    #
    #     model_embeds = self.model.get_decoder_input_embeddings()
    #
    #     if new_num_tokens is None:
    #         return model_embeds
    #
    #     # Update base model and current model config
    #     self.config.decoder_vocab_size = new_num_tokens
    #
    #     # Tie weights again if needed
    #     self.tie_weights()
    #
    #     self._resize_final_logits_bias(new_num_tokens)
    #
    #     return model_embeds
    #
    # def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
    #     old_num_tokens = self.final_logits_bias.shape[-1]
    #     if new_num_tokens <= old_num_tokens:
    #         new_bias = self.final_logits_bias[:, :new_num_tokens]
    #     else:
    #         extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
    #         new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    #     self.register_buffer("final_logits_bias", new_bias)
    #
    # def get_output_embeddings(self):
    #     return self.lm_head
    #
    # def set_output_embeddings(self, new_embeddings: nn.Embedding):
    #     self.lm_head = new_embeddings
    #
    # def tie_weights(self):
    #     """
    #     Tie the weights between the input embeddings and the output embeddings.
    #
    #     If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
    #     weights instead.
    #     """
    #     output_embeddings = self.get_output_embeddings()
    #     if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
    #         # if embeddings are shared this will return shared embeddings otherwise decoder embed_tokens
    #         word_embeddings = self.get_decoder().get_input_embeddings()
    #         self._tie_or_clone_weights(output_embeddings, word_embeddings)
    #
    #     if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
    #         if hasattr(self, self.base_model_prefix):
    #             self = getattr(self, self.base_model_prefix)
    #         self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)
    #
    #     for module in self.modules():
    #         if hasattr(module, "_tie_weights"):
    #             module._tie_weights()
    #
    # @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_head_disturbance_mask: Optional[torch.Tensor] = None,
            encoder_head_disturbance_value: float = 0.5,
            decoder_self_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            decoder_self_attention_head_disturbance_value: float = 0.5,
            decoder_cross_attention_head_disturbance_mask: Optional[torch.Tensor] = None,
            decoder_cross_attention_head_disturbance_value: float = 0.5,
            encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # Ignored arguments
            src_context_idx=None,
            src_phrase_idx=None,
            tgt_context_idx=None,
            tgt_phrase_idx=None,
    ) -> AdjustableSeq2SeqLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_head_disturbance_mask=encoder_head_disturbance_mask,
            encoder_head_disturbance_value=encoder_head_disturbance_value,
            decoder_self_attention_head_disturbance_mask=decoder_self_attention_head_disturbance_mask,
            decoder_self_attention_head_disturbance_value=decoder_self_attention_head_disturbance_value,
            decoder_cross_attention_head_disturbance_mask=decoder_cross_attention_head_disturbance_mask,
            decoder_cross_attention_head_disturbance_value=decoder_cross_attention_head_disturbance_value,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return AdjustableSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            decoder_pre_attentions=outputs.decoder_pre_attentions,
            cross_pre_attentions=outputs.cross_pre_attentions,
            encoder_pre_attentions=outputs.encoder_pre_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
        **kwargs,
    ) -> Dict:
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        returns = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if 'decoder_attention_mask' in kwargs:
            returns['decoder_attention_mask'] = kwargs['decoder_attention_mask']

        return returns

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def adjust_logits_during_generation(self, logits, cur_len):
        logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
        return logits

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
