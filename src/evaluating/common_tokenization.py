import torch

from data import concatenate_current_and_context


def get_context(context, context_size):
    if context is None or len(context) == 0:
        return [], 0

    batched = isinstance(context, list) and isinstance(context[0], list)

    if not batched:
        context = [context]

    effective_context = []
    for c in context:
        if c is None or len(c) == 0 or context_size is None or context_size == 0:
            c = []
        else:
            max_context = min(len(c), context_size)
            c = c[-max_context:]
        effective_context.append(c)

    if not batched:
        return effective_context[0], len(effective_context[0])

    return effective_context, [len(c) for c in effective_context]


def tokenize(tokenizer, device, text, is_target=False, max_length=None):
    # text = text.encode('utf-8')
    if is_target:
        return tokenizer(text_target=text,
                         max_length=max_length,
                         truncation=True,
                         padding=True,
                         return_token_type_ids=False,
                         return_tensors='pt', ).to(device)
    else:
        return tokenizer(text=text,
                         max_length=max_length,
                         truncation=True,
                         padding=True,
                         return_token_type_ids=False,
                         return_tensors='pt', ).to(device)


def tokenize_with_context(tokenizer, current, context, context_size, is_target: bool, max_length=None):
    context, effective_context_size = get_context(context, context_size)
    all = concatenate_current_and_context(current, context, tokenizer.sep_token)
    all_tokenized = tokenize(tokenizer, 'cpu', all, is_target, max_length=max_length)
    return all_tokenized, effective_context_size


def tokenize_all_with_context(tokenizer,
                              sources, targets,
                              sources_context, targets_context,
                              source_context_size, target_context_size,
                              max_length=None):
    sources_context = [get_context(context, source_context_size) for context in sources_context]
    targets_context = [get_context(context, target_context_size) for context in targets_context]
    sources_context_sizes = [c[1] for c in sources_context]
    targets_context_sizes = [c[1] for c in targets_context]
    sources_context = [c[0] for c in sources_context]
    targets_context = [c[0] for c in targets_context]

    sep_token = tokenizer.sep_token
    all_sources = concatenate_current_and_context(sources, sources_context, sep_token)
    all_targets = concatenate_current_and_context(targets, targets_context, sep_token)
    all_tokenized = tokenizer(text=all_sources, text_target=all_targets,
                              max_length=max_length,
                              padding=True, truncation=True,
                              return_tensors='pt',
                              return_token_type_ids=False,
                              )
    return all_tokenized, sources_context_sizes, targets_context_sizes


def analyze_tokenized(all_tokens, context_size, context_phrase_distance,
                      phrase_tokenized, context_phrase_tokenized,
                      sep_token):
    num_all_tokens = all_tokens.shape[0]
    num_phrase_tokens = phrase_tokenized.shape[0]

    if context_phrase_tokenized is not None:
        num_context_phrase_tokens = context_phrase_tokenized.shape[0]
    else:
        num_context_phrase_tokens = -1

    # covers the case where the separator token is not used but the context phrase is in the context window
    if sep_token is None and context_size >= context_phrase_distance:
        current_sentence = 0
        context_phrase_distance = 0
    else:
        current_sentence = context_size

    phrase_indices = []
    # inner_phrase_indices = []
    context_phrase_indices = []
    # inner_context_phrase_indices = []
    for i in range(num_all_tokens):
        token = all_tokens[i]
        if token == sep_token:
            current_sentence -= 1

        if current_sentence == 0:
            if (i + num_phrase_tokens <= num_all_tokens
                    and torch.all(all_tokens[i:i + num_phrase_tokens] == phrase_tokenized)):
                phrase_indices.append(list([i + j for j in range(num_phrase_tokens)]))

        # if current_sentence == 0:
        #     if (i + num_inner_phrase_tokens <= num_all_tokens
        #             and torch.all(all_tokens[i:i + num_inner_phrase_tokens] == inner_phrase_tokenized)):
        #         inner_phrase_indices.append(list([i + j for j in range(num_inner_phrase_tokens)]))

        if current_sentence == context_phrase_distance and context_phrase_tokenized is not None:
            if i + num_context_phrase_tokens <= num_all_tokens:
                if torch.all(all_tokens[i:i + num_context_phrase_tokens] == context_phrase_tokenized):
                    context_phrase_indices.append(list([i + j for j in range(num_context_phrase_tokens)]))

        # if current_sentence == context_phrase_distance and inner_context_phrase_tokenized is not None:
        #     if i + num_inner_context_phrase_tokens <= num_all_tokens:
        #         if torch.all(all_tokens[i:i + num_inner_context_phrase_tokens] == inner_context_phrase_tokenized):
        #             inner_context_phrase_indices.append(list([i + j for j in range(num_inner_context_phrase_tokens)]))

    # if len(phrase_indices) == 0:
    #     if len(inner_phrase_indices) > 0:
    #         phrase_indices = inner_phrase_indices
    #
    # if len(context_phrase_indices) == 0 and len(inner_context_phrase_indices) > 0:
    #     context_phrase_indices = inner_context_phrase_indices

    return phrase_indices, context_phrase_indices


def strip_tokenized(tokenizer, tokens):
    special_ids = tokenizer.all_special_ids
    start_id = 0
    end_id = tokens.shape[0]
    for i in range(tokens.shape[0]):
        if tokens[i] not in special_ids:
            start_id = i
            break
    for i in range(tokens.shape[0] - 1, -1, -1):
        if tokens[i] not in special_ids:
            end_id = i + 1
            break
    return tokens[start_id:end_id]


def tokenize_with_context_and_find_phrase(tokenizer, current, context, context_size,
                                          phrase, context_phrase, context_phrase_distance,
                                          is_target: bool, max_length=None, consider_upper_phrases=False):
    all_tokenized, effective_context_size = tokenize_with_context(tokenizer,
                                                                  current, context,
                                                                  context_size, is_target,
                                                                  max_length=max_length)
    all_tokens = all_tokenized['input_ids'][0]

    phrases_tokenized = tokenize_plausible_phrases(tokenizer, phrase, is_target, consider_upper_phrases)
    context_phrases_tokenized = tokenize_plausible_phrases(tokenizer, context_phrase, is_target, consider_upper_phrases)

    phrase_indices_new, context_phrase_indices_new = analyze_tokens_cascading(
        all_tokens, effective_context_size, context_phrase_distance,
        phrases_tokenized,
        context_phrases_tokenized if context_size >= context_phrase_distance else None,
        tokenizer.sep_token_id
    )

    # phrase_tokenized = tokenize(tokenizer, 'cpu', phrase, is_target)['input_ids'][0, :-1]
    phrase_tokenized = tokenize(tokenizer, 'cpu', phrase, is_target)['input_ids'][0]
    phrase_tokenized = strip_tokenized(tokenizer, phrase_tokenized)

    if context_phrase is not None:
        # context_phrase_tokenized = tokenize(tokenizer, 'cpu', context_phrase, is_target)['input_ids'][0, :-1]
        context_phrase_tokenized = tokenize(tokenizer, 'cpu', context_phrase, is_target)['input_ids'][0]
        context_phrase_tokenized = strip_tokenized(tokenizer, context_phrase_tokenized)
    else:
        context_phrase_tokenized = None

    phrase_indices, context_phrase_indices = analyze_tokenized(
        all_tokens,
        effective_context_size, context_phrase_distance,
        phrase_tokenized, context_phrase_tokenized,
        # inner_phrase_tokenized, inner_context_phrase_tokenized,
        tokenizer.sep_token_id
    )

    if len(phrase_indices) == 0 or len(context_phrase_indices) == 0:
        # covers the case where the phrase is following a symbol (e.g. "'")
        # inner_phrase_tokenized = tokenize(tokenizer, 'cpu', f"'{phrase}", is_target)['input_ids'][0, 1:-1]
        inner_phrase_tokenized = tokenize(tokenizer, 'cpu', f"'{phrase}", is_target)['input_ids'][0]
        inner_phrase_tokenized = strip_tokenized(tokenizer, inner_phrase_tokenized)
        inner_phrase_tokenized = inner_phrase_tokenized[1:]
        if context_phrase is not None:
            # inner_context_phrase_tokenized = tokenize(tokenizer, 'cpu', f"'{context_phrase}", is_target)['input_ids'][0, 1:-1]
            inner_context_phrase_tokenized = tokenize(tokenizer, 'cpu', f"'{context_phrase}", is_target)['input_ids'][0]
            inner_context_phrase_tokenized = strip_tokenized(tokenizer, inner_context_phrase_tokenized)
            inner_context_phrase_tokenized = inner_context_phrase_tokenized[1:]
        else:
            inner_context_phrase_tokenized = None

        inner_phrase_indices, inner_context_phrase_indices = analyze_tokenized(
            all_tokens,
            effective_context_size, context_phrase_distance,
            # phrase_tokenized, context_phrase_tokenized,
            inner_phrase_tokenized, inner_context_phrase_tokenized,
            tokenizer.sep_token_id
        )

        if len(phrase_indices) == 0 and len(inner_phrase_indices) > 0:
            phrase_indices = inner_phrase_indices

        if len(context_phrase_indices) == 0 and len(inner_context_phrase_indices) > 0:
            context_phrase_indices = inner_context_phrase_indices

    # if phrase_indices_new != phrase_indices:
    #     print('Warning: Phrase indices do not match')
    #     print(phrase_indices_new)
    #     print(phrase_indices)
    #     print('current:', current)
    #     print('context:', context)
    #     print('phrase:', phrase)
    #     print()
    #
    # if context_phrase_indices_new != context_phrase_indices:
    #     print('Warning: Context phrase indices do not match')
    #     print(context_phrase_indices_new)
    #     print(context_phrase_indices)
    #     print('current:', current)
    #     print('context:', context)
    #     print('context_phrase:', context_phrase)
    #     print('effective_context_size:', effective_context_size)
    #     print('context_phrase_distance:', context_phrase_distance)
    #     print()

    return all_tokenized, phrase_indices_new, context_phrase_indices_new


def tokenize_plausible_phrases(tokenizer, phrase, is_target, add_upper_phrase):
    if phrase is None:
        return [None, None, None, None]

    # upper_first = lambda p: p[0].upper() + p[1:]
    upper_phrase = phrase[0].upper() + phrase[1:]

    all_tokenized = []

    tokenized = tokenize(tokenizer, 'cpu', phrase, is_target)['input_ids'][0]
    tokenized = strip_tokenized(tokenizer, tokenized)
    all_tokenized.append(tokenized)

    if add_upper_phrase:
        tokenized = tokenize(tokenizer, 'cpu', upper_phrase, is_target)['input_ids'][0]
        tokenized = strip_tokenized(tokenizer, tokenized)
        all_tokenized.append(tokenized)

    tokenized = tokenize(tokenizer, 'cpu', f"'{phrase}", is_target)['input_ids'][0]
    tokenized = strip_tokenized(tokenizer, tokenized)
    tokenized = tokenized[1:]
    all_tokenized.append(tokenized)

    if add_upper_phrase:
        tokenized = tokenize(tokenizer, 'cpu', f"'{upper_phrase}", is_target)['input_ids'][0]
        tokenized = strip_tokenized(tokenizer, tokenized)
        tokenized = tokenized[1:]
        all_tokenized.append(tokenized)

    # tokenized.append(tokenize(tokenizer, 'cpu', phrase, is_target)['input_ids'][0, :-1])
    # tokenized.append(tokenize(tokenizer, 'cpu', upper_phrase, is_target)['input_ids'][0, :-1])
    # tokenized.append(tokenize(tokenizer, 'cpu', f"'{phrase}", is_target)['input_ids'][0, 1:-1])
    # tokenized.append(tokenize(tokenizer, 'cpu', f"'{upper_phrase}", is_target)['input_ids'][0, 1:-1])

    return all_tokenized


def analyze_tokens_cascading(all_tokens, context_size, context_phrase_distance,
                             phrases_tokenized, context_phrases_tokenized,
                             sep_token):
    phrase_indices = []
    context_phrase_indices = []

    def all_empty(l):
        return all([e is None or len(e) == 0 for e in l])

    if context_phrases_tokenized is None:
        context_phrases_tokenized = [None for _ in phrases_tokenized]

    for phrase_tokenized, context_phrase_tokenized in zip(phrases_tokenized, context_phrases_tokenized):
        potential_phrase_indices, potential_context_phrase_indices = analyze_tokenized(
            all_tokens, context_size, context_phrase_distance,
            phrase_tokenized, context_phrase_tokenized,
            sep_token
        )

        if len(potential_phrase_indices) > 0:
            if len(phrase_indices) == 0 or all_empty(phrase_indices):
                phrase_indices.extend(potential_phrase_indices)

        if len(potential_context_phrase_indices) > 0:
            if len(context_phrase_indices) == 0 or all_empty(context_phrase_indices):
                context_phrase_indices.extend(potential_context_phrase_indices)

        # if len(phrase_indices) > 0 and len(context_phrase_indices) > 0:
        #     break

    # if len(phrase_indices) == 0 or len(context_phrase_indices) == 0:
    #     print("Warning: No phrase found in the tokens")

    phrase_indices = [p for p in phrase_indices if p is not None and len(p) > 0]
    context_phrase_indices = [p for p in context_phrase_indices if p is not None and len(p) > 0]

    phrase_indices.sort(key=lambda p: p[0])
    context_phrase_indices.sort(key=lambda p: p[0])

    return phrase_indices, context_phrase_indices


def tokenize_and_analyze(tokenizer, sources, targets,
                         sources_context, targets_context,
                         source_context_size, target_context_size,
                         source_phrases, target_phrases,
                         source_context_phrases, target_context_phrases,
                         context_distances,
                         max_length=None,
                         consider_upper_phrases=False):
    (
        all_tokenized,
        sources_context_sizes,
        targets_context_sizes
    ) = tokenize_all_with_context(tokenizer, sources, targets,
                                  sources_context, targets_context,
                                  source_context_size, target_context_size,
                                  max_length=max_length)

    all_source_tokens = all_tokenized['input_ids']
    all_target_tokens = all_tokenized['labels']

    source_phrases_tokenized = [tokenize_plausible_phrases(tokenizer, p, False, consider_upper_phrases)
                                for p in source_phrases]
    target_phrases_tokenized = [tokenize_plausible_phrases(tokenizer, p, True, consider_upper_phrases)
                                for p in target_phrases]
    source_context_phrases_tokenized = [tokenize_plausible_phrases(tokenizer, p, False, consider_upper_phrases)
                                        for p in source_context_phrases]
    target_context_phrases_tokenized = [tokenize_plausible_phrases(tokenizer, p, True, consider_upper_phrases)
                                        for p in target_context_phrases]

    sep_token = tokenizer.sep_token_id

    sources_phrase_indices = []
    sources_context_phrase_indices = []
    targets_phrase_indices = []
    targets_context_phrase_indices = []

    for i in range(all_source_tokens.shape[0]):
        source_context_phrase_tokenized = source_context_phrases_tokenized[i]
        if source_context_size < context_distances[i]:
            source_context_phrase_tokenized = None
        source_phrase_indices, source_context_phrase_indices = analyze_tokens_cascading(
            all_source_tokens[i], sources_context_sizes[i], context_distances[i],
            source_phrases_tokenized[i],
            source_context_phrase_tokenized,
            sep_token
        )

        target_context_phrase_tokenized = target_context_phrases_tokenized[i]
        if target_context_size < context_distances[i]:
            target_context_phrase_tokenized = None
        target_phrase_indices, target_context_phrase_indices = analyze_tokens_cascading(
            all_target_tokens[i], targets_context_sizes[i], context_distances[i],
            target_phrases_tokenized[i],
            target_context_phrase_tokenized,
            sep_token
        )

        if len(source_phrase_indices) == 0:
            source = sources[i]
            context = sources_context[i]
            source_phrase = source_phrases[i]
            source_context_phrase = source_context_phrases[i]
            source_tokens = all_source_tokens[i]
            source_context_size = sources_context_sizes[i]
            context_disatnce = context_distances[i]
            source_phrase_tokens = source_phrases_tokenized[i]
            source_phrases_tokens = [tokenizer.convert_ids_to_tokens(p) for p in source_phrase_tokens]
            source_contexts = source_context_phrases_tokenized[i]
            source_contexts_tokens = [tokenizer.convert_ids_to_tokens(p) for p in source_contexts]
            print("Warning: No phrase found in the source tokens")

        if len(target_phrase_indices) == 0:
            target = targets[i]
            context = targets_context[i]
            target_phrase = target_phrases[i]
            target_context_phrase = target_context_phrases[i]
            target_tokens = all_target_tokens[i]
            target_context_size = targets_context_sizes[i]
            context_disatnce = context_distances[i]
            target_phrase_tokenized = target_phrases_tokenized[i]
            target_phrases_tokens = [tokenizer.convert_ids_to_tokens(p) for p in target_phrase_tokenized]
            target_contexts = target_context_phrases_tokenized[i]
            target_contexts_tokens = [tokenizer.convert_ids_to_tokens(p) for p in target_contexts]
            print("Warning: No phrase found in the target tokens")

        last_source_phrase = source_phrase_indices[-1] if len(source_phrase_indices) > 0 else None
        first_source_context = source_context_phrase_indices[0] if len(source_context_phrase_indices) > 0 else None
        last_target_phrase = target_phrase_indices[-1] if len(target_phrase_indices) > 0 else None
        first_target_context = target_context_phrase_indices[0] if len(target_context_phrase_indices) > 0 else None

        if last_source_phrase is not None and first_source_context is not None and last_source_phrase[0] < \
                first_source_context[0]:
            source = sources[i]
            context = sources_context[i]
            source_phrase = source_phrases[i]
            source_context_phrase = source_context_phrases[i]
            source_tokens = all_source_tokens[i]
            source_context_size = sources_context_sizes[i]
            context_disatnce = context_distances[i]
            source_phrase_tokens = source_phrases_tokenized[i]
            source_phrases_tokens = [tokenizer.convert_ids_to_tokens(p) for p in source_phrase_tokens]
            source_contexts = source_context_phrases_tokenized[i]
            source_contexts_tokens = [tokenizer.convert_ids_to_tokens(p) for p in source_contexts]
            print('Warning: Source phrase is before the source context phrase')

        if last_target_phrase is not None and first_target_context is not None and last_target_phrase[0] < \
                first_target_context[0]:
            target = targets[i]
            context = targets_context[i]
            target_phrase = target_phrases[i]
            target_context_phrase = target_context_phrases[i]
            target_tokens = all_target_tokens[i]
            target_context_size = targets_context_sizes[i]
            context_disatnce = context_distances[i]
            target_phrase_tokenized = target_phrases_tokenized[i]
            target_phrases_tokens = [tokenizer.convert_ids_to_tokens(p) for p in target_phrase_tokenized]
            target_contexts = target_context_phrases_tokenized[i]
            target_contexts_tokens = [tokenizer.convert_ids_to_tokens(p) for p in target_contexts]
            target_tokens_detokenized = tokenizer.convert_ids_to_tokens(target_tokens)
            target_tokens_detokenized_len = len(tokenizer.convert_ids_to_tokens(target_tokens))
            print('Warning: Target phrase is before the target context phrase')

        sources_phrase_indices.append(source_phrase_indices)
        sources_context_phrase_indices.append(source_context_phrase_indices)
        targets_phrase_indices.append(target_phrase_indices)
        targets_context_phrase_indices.append(target_context_phrase_indices)

    return (all_tokenized,
            sources_phrase_indices,
            sources_context_phrase_indices,
            targets_phrase_indices,
            targets_context_phrase_indices)
