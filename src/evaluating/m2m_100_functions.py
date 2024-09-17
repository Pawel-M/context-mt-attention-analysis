from typing import List

import torch
from transformers import BatchEncoding
from transformers.utils import logging

from data import concatenate_current_and_context
from data.contrapro import DataPoint
from evaluating.utils import (
    get_sentence_score,
)

from evaluating.common_functions import (
    disable_attention_heads,
    enable_attention_heads,
    create_modification_mask, HeadDisabler, combine_modification_masks,
)

from evaluating.common_tokenization import (
    tokenize,
    tokenize_with_context,
    tokenize_with_context_and_find_phrase
)


def generate_translation_raw(model, tokenizer, device,
                             source_context_size, target_context_size,
                             source, source_context, target_context,
                             num_beams=5, max_len=300):
    tokenized_src, _ = tokenize_with_context(tokenizer,
                                             source, source_context, source_context_size,
                                             is_target=False)
    tokenized_src = tokenized_src.to(device)

    if target_context_size is not None and target_context_size > 0:
        empty_target = ''
        if type(source) is list:
            empty_target = [''] * len(source)
        tokenized_tgt_context, _ = tokenize_with_context(tokenizer,
                                                         empty_target, target_context, target_context_size,
                                                         is_target=True)
        tokenized_tgt_context = tokenized_tgt_context.to(device)
        tokenized_tgt_context = tokenized_tgt_context['input_ids']
        tokenized_tgt_context = torch.where(tokenized_tgt_context == tokenizer.eos_token_id,
                                            tokenizer.pad_token_id, tokenized_tgt_context)
        # tokenized_tgt_context = tokenized_tgt_context['input_ids'][..., :-1]
        if tokenized_tgt_context.shape[1] == 0:
            tokenized_tgt_context = None
        else:
            tokenized_attention_mask = tokenized_tgt_context != tokenizer.pad_token_id
            tokenized_src['decoder_attention_mask'] = tokenized_attention_mask

        forced_bos_token_id = None
    else:
        tokenized_tgt_context = None
        forced_bos_token_id = tokenizer.lang_code_to_id[tokenizer.tgt_lang]

    generated_tokens = model.generate(
        **tokenized_src,
        decoder_input_ids=tokenized_tgt_context,
        forced_bos_token_id=forced_bos_token_id,
        num_beams=num_beams, do_sample=False, max_length=max_len, output_scores=True,
        min_new_tokens=1,
    )

    if tokenized_tgt_context is not None:
        generated_tokens = generated_tokens[..., tokenized_tgt_context.shape[1] + 1:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded


def generate_translation(model, tokenizer, device,
                         source_context_size, target_context_size,
                         data: DataPoint, num_beams=5, max_len=300):
    decoded = generate_translation_raw(model, tokenizer, device,
                                       source_context_size, target_context_size,
                                       data.source, data.source_context, data.target_context,
                                       num_beams=num_beams, max_len=max_len)

    return decoded[0]
