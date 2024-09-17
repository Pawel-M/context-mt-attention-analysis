from typing import List, Optional

import torch
from transformers import BatchEncoding

from data.contrapro import DataPoint

from evaluating.utils import (
    get_sentence_score,
)

from evaluating.common_tokenization import (
    tokenize,
    tokenize_with_context,
    tokenize_with_context_and_find_phrase
)


class HeadDisabler(object):
    def __init__(self, module, attention_name, layers_heads, disable_whole_heads, token_ids):
        self.module = module
        self.attention_name = attention_name
        self.layers_heads = layers_heads
        self.disable_whole_heads = disable_whole_heads
        self.token_ids = token_ids

    def __enter__(self):
        disable_attention_heads(self.module, self.attention_name, self.layers_heads, self.disable_whole_heads,
                                self.token_ids)

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_attention_heads(self.module, self.attention_name, self.layers_heads)


def disable_attention_heads(module, attention_name, layers_heads, disable_whole_heads, token_ids):
    if layers_heads is not None:
        for layer, head in layers_heads:
            attn = getattr(module.layers[layer], attention_name)
            attn.disabled_heads = ([] if attn.disabled_heads is None else attn.disabled_heads) + [head]
            if not disable_whole_heads and token_ids is not None:
                attn.disabled_heads_tokens = token_ids


def enable_attention_heads(module, attention_name, layers_heads):
    if layers_heads is not None:
        for layer, head in layers_heads:
            attn = getattr(module.layers[layer], attention_name)
            attn.disabled_heads = None
            attn.disabled_heads_tokens = None


def create_modification_mask(device, num_layers, num_heads, num_src_tokens, num_tgt_tokens,
                             modify_layers_heads, modify_src_tokens, modify_tgt_tokens):
    if modify_layers_heads is None:
        return None
    if not modify_src_tokens or not modify_tgt_tokens:
        return None

    head_disturbance_mask = torch.zeros(
        (num_layers, 1, num_heads, num_src_tokens, num_tgt_tokens),
        dtype=torch.float, device=device)

    for layer_head in modify_layers_heads:
        modify_layers = layer_head[0]
        modify_heads = layer_head[1]
        for src_token in modify_src_tokens:
            head_disturbance_mask[modify_layers, :, modify_heads, src_token, modify_tgt_tokens] = 1

    return head_disturbance_mask


def combine_modification_masks(*masks: List[Optional[torch.Tensor]]):
    if all(mask is None for mask in masks):
        return None

    non_none_masks = [mask for mask in masks if mask is not None]
    return torch.stack(non_none_masks, dim=0).sum(dim=0)


def analyze_contrastive_inputs(tokenizer, device,
                               source_context_size, target_context_size, data: DataPoint,
                               consider_upper_phrases=False):
    tokenized_src, src_phrase_indices, src_context_phrase_indices = tokenize_with_context_and_find_phrase(
        tokenizer, data.source, data.source_context, source_context_size,
        data.source_phrase, data.source_context_phrase, data.context_distance,
        is_target=False, consider_upper_phrases=consider_upper_phrases)
    tokenized_src = tokenized_src.to(device)

    tgts_analysis = [tokenize_with_context_and_find_phrase(
        tokenizer, target, data.target_context, target_context_size,
        target_phrase, data.target_context_phrase, data.context_distance,
        is_target=True, consider_upper_phrases=consider_upper_phrases)
        for target, target_phrase in zip(data.targets, data.targets_phrases)]
    tgt_phrases_indices = [ta[1] for ta in tgts_analysis]
    tgt_context_phrases_indices = [ta[2] for ta in tgts_analysis]
    tokenized_tgts = [ta[0].to(device) for ta in tgts_analysis]

    return (tokenized_src,
            src_phrase_indices,
            src_context_phrase_indices,
            tokenized_tgts,
            tgt_phrases_indices,
            tgt_context_phrases_indices)


def collate_inputs(tokenizer, inputs: List[BatchEncoding]):
    if len(inputs) < 2:
        return inputs

    max_len = max([e['input_ids'].shape[1] for e in inputs])
    padded_input_ids = []
    padded_attention_mask = []
    for e in inputs:
        input_ids = e['input_ids']
        attention_mask = e['attention_mask']
        num_tokens = input_ids.shape[1]
        padding = max_len - num_tokens
        padding_ids = torch.ones((1, padding), dtype=torch.long) * tokenizer.pad_token_id
        padded_input_ids.append(torch.cat([input_ids, padding_ids], dim=-1))
        padded_attention_mask.append(torch.cat([attention_mask, torch.zeros((1, padding), dtype=torch.long)], dim=-1))
    input_ids = torch.cat(padded_input_ids, dim=0)
    attention_mask = torch.cat(padded_attention_mask, dim=0)
    return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask})


def score_contrastive(model, tokenizer, device,
                      source_context_size, target_context_size,
                      data: DataPoint,
                      return_attentions=False,
                      consider_upper_phrases=False):
    (
        tokenized_src,
        src_phrase_indices,
        src_context_phrase_indices,
        tokenized_tgts,
        tgt_phrases_indices,
        tgt_context_phrases_indices
    ) = analyze_contrastive_inputs(tokenizer, device, source_context_size, target_context_size, data,
                                   consider_upper_phrases=consider_upper_phrases)

    pad_token_id = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    encoder = model.get_encoder()

    encoder_out = encoder(
        input_ids=tokenized_src['input_ids'],
        attention_mask=tokenized_src['attention_mask'],
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    self_attention = [sa.detach().cpu() for sa in encoder_out['attentions']] if return_attentions else []

    decoder = model.get_decoder()
    head = model.lm_head
    head_bias = model.final_logits_bias if hasattr(model, 'final_logits_bias') else None

    total_logprobs = []
    total_probs = []
    cross_attentions = []
    decoder_attentions = []
    tokens = [tokenizer.convert_ids_to_tokens(tokenized_src['input_ids'][0]), []]
    target_logits = []
    target_encoded_ids = []
    for tokenized_tgt in tokenized_tgts:
        encoded_tgt_ids = tokenized_tgt['input_ids']
        target_encoded_ids.append(encoded_tgt_ids)
        tokens[1].append(tokenizer.convert_ids_to_tokens(encoded_tgt_ids[0]))

        input_tgt_ids = torch.concat((pad_token_id, encoded_tgt_ids[:, :-1]), dim=-1)
        decoder_outputs = decoder(
            input_ids=input_tgt_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_out['last_hidden_state'],
            encoder_attention_mask=tokenized_src['attention_mask'],
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_hidden_state = decoder_outputs['last_hidden_state']
        if return_attentions:
            cross_attentions.append([ca.detach().cpu() for ca in decoder_outputs['cross_attentions']])
            decoder_attentions.append([da.detach().cpu() for da in decoder_outputs['attentions']])

        logits = head(decoder_hidden_state)
        if head_bias is not None:
            logits += head_bias

        target_logits.append(logits.detach().cpu())

        # print('argmax: ', torch.argmax(logits, dim=-1))
        total_logprob, total_prob = get_sentence_score(encoded_tgt_ids, logits)
        total_logprobs.append(total_logprob)
        total_probs.append(total_prob)

    target_encoded_ids = [encoded_tgt_ids.detach().cpu() for encoded_tgt_ids in target_encoded_ids]

    return_values = (
        total_logprobs,
        total_probs,
        tokens,
        target_encoded_ids,
        target_logits,
        src_phrase_indices,
        src_context_phrase_indices,
        tgt_phrases_indices,
        tgt_context_phrases_indices,
    )

    if return_attentions:
        return return_values + (self_attention, cross_attentions, decoder_attentions,)

    return return_values


def score_contrastive_disabling_heads(model, tokenizer, device,
                                      source_context_size, target_context_size,
                                      disable_heads_for_all_tokens,
                                      disabled_encoder_layers_heads,
                                      disabled_cross_attention_layer_heads,
                                      disabled_decoder_attention_layer_heads,
                                      data: DataPoint,
                                      return_attentions=False,
                                      consider_upper_phrases=False):
    (
        tokenized_src,
        src_phrase_indices,
        src_context_phrase_indices,
        tokenized_tgts,
        tgt_phrases_indices,
        tgt_context_phrases_indices
    ) = analyze_contrastive_inputs(tokenizer, device, source_context_size, target_context_size, data,
                                   consider_upper_phrases=consider_upper_phrases)

    pad_token_id = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    encoder = model.get_encoder()

    with HeadDisabler(encoder, 'self_attn',
                      disabled_encoder_layers_heads,
                      disable_heads_for_all_tokens,
                      src_phrase_indices[-1]):
        encoder_out = encoder(
            input_ids=tokenized_src['input_ids'],
            attention_mask=tokenized_src['attention_mask'],
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    self_attention = [sa.detach().cpu() for sa in encoder_out['attentions']] if return_attentions else []

    decoder = model.get_decoder()
    head = model.lm_head
    # check if model contains final_logits_bias
    head_bias = model.final_logits_bias if hasattr(model, 'final_logits_bias') else None

    total_logprobs = []
    total_probs = []
    cross_attentions = []
    decoder_attentions = []
    tokens = [tokenizer.convert_ids_to_tokens(tokenized_src['input_ids'][0]), []]
    target_logits = []
    target_encoded_ids = []
    for tgt_index, tokenized_tgt in enumerate(tokenized_tgts):
        encoded_tgt_ids = tokenized_tgt['input_ids']
        target_encoded_ids.append(encoded_tgt_ids)
        tokens[1].append(tokenizer.convert_ids_to_tokens(encoded_tgt_ids[0]))

        input_tgt_ids = torch.concat((pad_token_id, encoded_tgt_ids[:, :-1]), dim=-1)

        with HeadDisabler(decoder, 'encoder_attn',
                          disabled_cross_attention_layer_heads,
                          disable_heads_for_all_tokens,
                          tgt_phrases_indices[tgt_index][-1]), \
                HeadDisabler(decoder, 'self_attn',
                             disabled_decoder_attention_layer_heads,
                             disable_heads_for_all_tokens,
                             tgt_phrases_indices[tgt_index][
                                 -1]):
            decoder_outputs = decoder(
                input_ids=input_tgt_ids,
                attention_mask=None,
                encoder_hidden_states=encoder_out['last_hidden_state'],
                encoder_attention_mask=tokenized_src['attention_mask'],
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        decoder_hidden_state = decoder_outputs['last_hidden_state']
        if return_attentions:
            cross_attentions.append([ca.detach().cpu() for ca in decoder_outputs['cross_attentions']])
            decoder_attentions.append([da.detach().cpu() for da in decoder_outputs['attentions']])

        logits = head(decoder_hidden_state)
        if head_bias is not None:
            logits += head_bias

        target_logits.append(logits.detach().cpu())

        # print('argmax: ', torch.argmax(logits, dim=-1))
        total_logprob, total_prob = get_sentence_score(encoded_tgt_ids, logits)
        total_logprobs.append(total_logprob)
        total_probs.append(total_prob)

    target_encoded_ids = [encoded_tgt_ids.detach().cpu() for encoded_tgt_ids in target_encoded_ids]

    return_values = (
        total_logprobs,
        total_probs,
        tokens,
        target_encoded_ids,
        target_logits,
        src_phrase_indices,
        src_context_phrase_indices,
        tgt_phrases_indices,
        tgt_context_phrases_indices,
    )

    if return_attentions:
        return return_values + (self_attention, cross_attentions, decoder_attentions,)

    return return_values


def score_contrastive_modifying_attention(model, tokenizer, device,
                                          source_context_size, target_context_size,
                                          modify_to_value,
                                          encoder_layers_heads,
                                          phrase_cross_attention_layer_heads,
                                          context_cross_attention_layer_heads,
                                          decoder_attention_layer_heads,
                                          decoder_after_attention_layer_heads,
                                          data: DataPoint,
                                          return_attentions=False,
                                          consider_upper_phrases=False):
    (
        tokenized_src,
        src_phrase_indices,
        src_context_phrase_indices,
        tokenized_tgts,
        tgt_phrases_indices,
        tgt_context_phrases_indices
    ) = analyze_contrastive_inputs(tokenizer, device, source_context_size, target_context_size, data,
                                   consider_upper_phrases=consider_upper_phrases)

    pad_token_id = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    encoder = model.get_encoder()
    decoder = model.get_decoder()

    num_encoder_layers = len(encoder.layers)
    num_decoder_layers = len(decoder.layers)
    num_encoder_heads = encoder.layers[0].self_attn.num_heads
    num_decoder_heads = decoder.layers[0].self_attn.num_heads
    num_source_tokens = tokenized_src['input_ids'].size(1)

    current_src_phrase_indices = src_phrase_indices[-1]
    current_src_context_phrase_indices = src_context_phrase_indices
    if len(current_src_context_phrase_indices) > 0:
        current_src_context_phrase_indices = current_src_context_phrase_indices[0]

    encoder_disturbance_mask = create_modification_mask(
        device,
        num_encoder_layers, num_encoder_heads,
        num_source_tokens, num_source_tokens,
        encoder_layers_heads,
        current_src_phrase_indices, current_src_context_phrase_indices
    )

    encoder_out = encoder(
        input_ids=tokenized_src['input_ids'],
        attention_mask=tokenized_src['attention_mask'],
        head_disturbance_mask=encoder_disturbance_mask,
        head_disturbance_value=modify_to_value,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    self_attention = [sa.detach().cpu() for sa in encoder_out['attentions']] if return_attentions else []

    head = model.lm_head
    head_bias = model.final_logits_bias if hasattr(model, 'final_logits_bias') else None

    total_logprobs = []
    total_probs = []
    cross_attentions = []
    decoder_attentions = []
    tokens = [tokenizer.convert_ids_to_tokens(tokenized_src['input_ids'][0]), []]
    target_logits = []
    target_encoded_ids = []
    for tgt_index, tokenized_tgt in enumerate(tokenized_tgts):
        encoded_tgt_ids = tokenized_tgt['input_ids']
        target_encoded_ids.append(encoded_tgt_ids)
        tokens[1].append(tokenizer.convert_ids_to_tokens(encoded_tgt_ids[0]))
        input_tgt_ids = torch.concat((pad_token_id, encoded_tgt_ids[:, :-1]), dim=-1)
        num_target_tokens = input_tgt_ids.size(1)

        current_tgt_phrase_indices = tgt_phrases_indices[tgt_index][-1]
        current_tgt_context_phrase_indices = tgt_context_phrases_indices[tgt_index]
        if len(current_tgt_context_phrase_indices) > 0:
            current_tgt_context_phrase_indices = current_tgt_context_phrase_indices[0]

        context_cross_attention_disturbance_mask = create_modification_mask(
            input_tgt_ids.device,
            num_decoder_layers, num_decoder_heads,
            num_target_tokens, num_source_tokens,
            context_cross_attention_layer_heads,
            modify_src_tokens=current_tgt_phrase_indices,
            modify_tgt_tokens=current_src_context_phrase_indices
        )
        phrase_cross_attention_disturbance_mask = create_modification_mask(
            input_tgt_ids.device,
            num_decoder_layers, num_decoder_heads,
            num_target_tokens, num_source_tokens,
            phrase_cross_attention_layer_heads,
            modify_src_tokens=current_tgt_phrase_indices,
            modify_tgt_tokens=current_src_phrase_indices
        )
        cross_attention_disturbance_mask = combine_modification_masks(
            context_cross_attention_disturbance_mask, phrase_cross_attention_disturbance_mask)

        decoder_attention_disturbance_mask = create_modification_mask(
            input_tgt_ids.device,
            num_decoder_layers, num_decoder_heads,
            num_target_tokens, num_target_tokens,
            decoder_attention_layer_heads,
            modify_src_tokens=current_tgt_phrase_indices,
            modify_tgt_tokens=current_tgt_context_phrase_indices
        )

        decoder_after_attention_disturbance_mask = create_modification_mask(
            input_tgt_ids.device,
            num_decoder_layers, num_decoder_heads,
            num_target_tokens, num_target_tokens,
            decoder_after_attention_layer_heads,
            modify_src_tokens=current_tgt_phrase_indices,
            modify_tgt_tokens=[i + 1 for i in current_tgt_context_phrase_indices]
        )

        decoder_attention_disturbance_mask = combine_modification_masks(
            decoder_attention_disturbance_mask, decoder_after_attention_disturbance_mask)

        decoder_outputs = decoder(
            input_ids=input_tgt_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_out['last_hidden_state'],
            encoder_attention_mask=tokenized_src['attention_mask'],
            self_attention_head_disturbance_mask=decoder_attention_disturbance_mask,
            self_attention_head_disturbance_value=modify_to_value,
            cross_attention_head_disturbance_mask=cross_attention_disturbance_mask,
            cross_attention_head_disturbance_value=modify_to_value,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_hidden_state = decoder_outputs['last_hidden_state']
        if return_attentions:
            cross_attentions.append([ca.detach().cpu() for ca in decoder_outputs['cross_attentions']])
            decoder_attentions.append([da.detach().cpu() for da in decoder_outputs['attentions']])

        logits = head(decoder_hidden_state)
        if head_bias is not None:
            logits += head_bias

        target_logits.append(logits.detach().cpu())

        # print('argmax: ', torch.argmax(logits, dim=-1))
        total_logprob, total_prob = get_sentence_score(encoded_tgt_ids, logits)
        total_logprobs.append(total_logprob)
        total_probs.append(total_prob)

    target_encoded_ids = [encoded_tgt_ids.detach().cpu() for encoded_tgt_ids in target_encoded_ids]

    return_values = (
        total_logprobs,
        total_probs,
        tokens,
        target_encoded_ids,
        target_logits,
        src_phrase_indices,
        src_context_phrase_indices,
        tgt_phrases_indices,
        tgt_context_phrases_indices,
    )

    if return_attentions:
        return return_values + (self_attention, cross_attentions, decoder_attentions,)

    return return_values
