import torch


def get_sentence_score(encoded_ids, logits, attention_mask=None):
    total_logprob = 0
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    for i in range(1, logits.shape[1]):
        if attention_mask is not None and attention_mask[0, i] == 0:
            continue
        prob = logprobs[0, i, encoded_ids[0, i]]
        total_logprob += prob

    return total_logprob.detach().cpu(), torch.exp(total_logprob).detach().cpu()


def parse_heads_list(attention_layer_heads_list, return_empty_if_none=False):
    if attention_layer_heads_list is None or len(attention_layer_heads_list) == 0:
        return tuple() if return_empty_if_none else None

    if type(attention_layer_heads_list[0]) == list:
        heads_lists = []
        for attention_layer_heads in attention_layer_heads_list:
            heads_lists.append(parse_heads_list(attention_layer_heads, return_empty_if_none=True))

        return heads_lists

    heads = []
    for attention_layer_head in attention_layer_heads_list:
        if type(attention_layer_head) == str:
            if attention_layer_head[0] == '(':  # this is a tuple
                attention_layer_head = attention_layer_head[1:-1]
            attention, layer, head = attention_layer_head.split(',')
            assert attention in ('encoder', 'cross', 'decoder') \
                   or attention in ('encoder', 'phrase_cross', 'context_cross', 'decoder', 'decoder_after'), \
                f'Invalid attention type: {attention}'
            heads.append((attention, int(layer), int(head)))

    return heads


def generate_full_heads_list(num_layers, num_heads, attention_types=('encoder', 'cross', 'decoder')):
    heads = []
    for layer in range(num_layers):
        for head in range(num_heads):
            for attention_type in attention_types:
                heads.append((attention_type, layer, head))

    return heads
