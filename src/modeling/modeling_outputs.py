from dataclasses import dataclass
from typing import Optional, Tuple

import torch.utils.checkpoint
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


@dataclass
class AdjustableBaseModelOutput(BaseModelOutput):
    pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AdjustableBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AdjustableSeq2SeqModelOutput(Seq2SeqModelOutput):
    decoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AdjustableSeq2SeqLMOutput(Seq2SeqLMOutput):
    decoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None

# @dataclass
# class AdjustableSeq2SeqLMOutput(Seq2SeqLMOutput):
#     decoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
#     cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
#     encoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


# TODO: possibly remove the following
@dataclass
class HeadTuneBaseModelOutput(BaseModelOutput):
    pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HeadTuneBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HeadTuneSeq2SeqModelOutput(Seq2SeqModelOutput):
    decoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HeadTuneSeq2SeqLMOutput(Seq2SeqLMOutput):
    decoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pre_attentions: Optional[Tuple[torch.FloatTensor]] = None
