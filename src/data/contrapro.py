import dataclasses
import json
import os
import re
import string
from typing import List, Optional

import spacy
import tqdm

en_nlp = spacy.load('en_core_web_sm')
de_nlp = spacy.load('de_core_news_sm')


@dataclasses.dataclass
class DataPoint:
    source: str
    target: str
    targets: Optional[List[str]]
    source_context: List[str]
    target_context: List[str]
    source_phrase: str
    target_phrase: str
    targets_phrases: Optional[List[str]]
    source_context_phrase: str
    source_context_phrase_id: Optional[int]
    target_context_phrase: Optional[str]
    context_distance: int
    src_phrase_indices: Optional[List[int]]
    src_context_phrase_indices: Optional[List[int]]
    tgt_phrases_indices: Optional[List[int]]
    tgt_context_phrases_indices: Optional[List[int]]
    data: dict


def load_contrapro(dir, filter_context_size=False, limit_ids=None):
    contrapro_file = os.path.join(dir, 'contrapro.json')

    if '~' in contrapro_file:
        contrapro_file = os.path.expanduser(contrapro_file)

    with open(contrapro_file, 'r') as f:
        data_json = json.load(f)

    data_old = []
    data = []
    for d in data_json:
        if filter_context_size and d['ante distance'] != 0:
            continue

        tgts = [d['ref segment']]
        tgts.extend([e['contrastive'] for e in d['errors']])

        data_old.append((d['src segment'], tgts, d))
        data.append(DataPoint(
            source=d['src segment'],
            target=d['ref segment'],
            targets=[d['ref segment']] + [error['contrastive'] for error in d['errors']],
            source_context=[],
            target_context=[],
            source_phrase=d['src pronoun'],
            target_phrase=d['ref pronoun'],
            targets_phrases=[d['ref pronoun']] + [error['replacement'] for error in d['errors']],
            source_context_phrase=d['src ante head'],
            source_context_phrase_id=d['src ante head id'],
            target_context_phrase=d['ref ante head'],
            context_distance=d['ante distance'],
            data=d,
            src_phrase_indices=None,
            src_context_phrase_indices=None,
            tgt_phrases_indices=None,
            tgt_context_phrases_indices=None,
        ))

    if limit_ids is not None:
        limited_data_old = []
        for i, d in enumerate(data_old):
            if i in limit_ids:
                limited_data_old.append(d)
        data_old = limited_data_old

        limited_data = []
        for i, d in enumerate(data):
            if i in limit_ids:
                limited_data.append(d)
        data = limited_data

    return data


def load_contrapro_with_context(dir, context_size, filter_context_size=False, limit_ids=None, use_json_lines=True):
    print(f'Loading data from "{dir}"...')
    print('use_json_lines:', use_json_lines)

    working_context_size = context_size if context_size is not None else 1
    working_context_size = max(working_context_size, 1)

    if '~' in dir:
        dir = os.path.expanduser(dir)

    contrapro_file = os.path.join(dir, 'contrapro.json')
    context_dir = os.path.join(dir, f'ctx{working_context_size}')
    src_file = os.path.join(context_dir, 'contrapro.text.en')
    tgt_file = os.path.join(context_dir, 'contrapro.text.de')
    src_context_file = os.path.join(context_dir, 'contrapro.context.en')
    tgt_context_file = os.path.join(context_dir, 'contrapro.context.de')

    with open(contrapro_file, 'r') as f:
        data_json = json.load(f)

    # data_old = []
    data = []
    context_id = 0

    with open(src_file, 'r') as f:
        src_lines = f.readlines()

    with open(tgt_file, 'r') as f:
        tgt_lines = f.readlines()

    with open(src_context_file, 'r') as f:
        src_context_lines = f.readlines()

    with open(tgt_context_file, 'r') as f:
        tgt_context_lines = f.readlines()

    sources_mismatched = 0
    targets_mismatched = 0
    contrastives_mismatched = 0
    for i, d in enumerate(data_json):

        context_start_line = context_id * working_context_size
        # src_context = src_context_lines[context_start_line:context_start_line + working_context_size]
        # tgt_context = tgt_context_lines[context_start_line:context_start_line + working_context_size]
        if context_size is not None and context_size > 0:
            src_context = src_context_lines[context_start_line:context_start_line + working_context_size]
            tgt_context = tgt_context_lines[context_start_line:context_start_line + working_context_size]
        else:
            src_context = []
            tgt_context = []

        # remove newline symbols
        src_context = [c.strip() for c in src_context]
        tgt_context = [c.strip() for c in tgt_context]
        if filter_context_size and context_size is not None and d['ante distance'] > context_size:
            context_id += len([d['errors']]) + 1
            continue

        # data_old.append((d['src segment'], tgts, src_context, tgt_context, d))

        # tgts = [d['ref segment']]
        # contrastive = [e['contrastive'] for e in d['errors']]
        # tgts.extend(contrastive)

        if use_json_lines:
            src_line = d['src segment'].strip()
            tgt_line = d['ref segment'].strip()
            contrastive = [e['contrastive'].strip() for e in d['errors']]
            tgts = [tgt_line] + contrastive
        else:
            src_line = src_lines[context_id].strip()
            tgt_line = tgt_lines[context_id].strip()
            contrastive = [e['contrastive'].strip() for e in d['errors']]
            tgts = [tgt_line] + [tgt_lines[context_id + i + 1].strip() for i in range(len(contrastive))]

        if d['src segment'].strip() != src_line:
            sources_mismatched += 1
            # d_src = d['src segment']
            # print(context_id, 'src')
            # print(d_src)
            # print(src_line)
            # print()

        if d['ref segment'].strip() != tgt_line:
            targets_mismatched += 1
            # d_tgt = d['ref segment']
            # print(context_id, 'tgt')
            # print(d_tgt)
            # print(tgt_line)
            # print()

        if d['errors'][0]['contrastive'].strip() != tgts[1]:
            contrastives_mismatched += 1
            # print(context_id, 'contrastive')
            # print(d['errors'][0]['contrastive'])
            # print(tgts[1])
            # print()

        data.append(DataPoint(
            # source=d['src segment'],
            # target=d['ref segment'],
            source=src_line,
            target=tgt_line,
            targets=tgts,
            source_context=src_context,
            target_context=tgt_context,
            source_phrase=d['src pronoun'],
            target_phrase=d['ref pronoun'],
            targets_phrases=[d['ref pronoun']] + [error['replacement'] for error in d['errors']],
            source_context_phrase=d['src ante head'],
            source_context_phrase_id=d['src ante head id'],
            target_context_phrase=d['ref ante head'],
            context_distance=d['ante distance'],
            data=d,
            src_phrase_indices=None,
            src_context_phrase_indices=None,
            tgt_phrases_indices=None,
            tgt_context_phrases_indices=None,
        ))

        context_id += len(tgts)

    print(f'Sources mismatched: {sources_mismatched}')
    print(f'Targets mismatched: {targets_mismatched}')
    print(f'Contrastives mismatched: {contrastives_mismatched}')

    if limit_ids is not None:
        # limited_data_old = []
        # for i, d in enumerate(data_old):
        #     if i in limit_ids:
        #         limited_data_old.append(d)
        # data_old = limited_data_old

        limited_data = []
        for i, d in enumerate(data):
            if i in limit_ids:
                limited_data.append(d)
        data = limited_data

    return data


def _split_sentence(sentence, lang):
    nlp = de_nlp if lang == 'de' else en_nlp
    doc = nlp(sentence)
    word_starts = [w.idx for w in doc]
    words = [w.text for w in doc]
    return word_starts, words


def _find_token_ids(word, sentence, offset_mapping, word_start=-1, use_byte_string=False):
    punctuation_whitespace = string.punctuation + string.whitespace
    if use_byte_string:
        word = word.encode('utf-8')
        sentence = sentence.encode('utf-8')
        punctuation_whitespace = punctuation_whitespace.encode('utf-8')

    word_len = len(word)
    matches = []
    distances = []
    for i in range(len(sentence) - word_len + 1):
        if sentence[i:i + word_len] == word:
            # is it the whole word?
            if (
                    ((i - 1 < 0) or sentence[i - 1] in punctuation_whitespace)
                    and ((i + word_len + 1 >= len(sentence))
                         or sentence[i + word_len] in punctuation_whitespace)
            ):
                matches.append((i, i + word_len))
                distances.append(abs(i - word_start))

    # didn't find the whole word (probably noise in the dataset), just find the occurrences
    if len(matches) == 0:
        for i in range(len(sentence) - word_len + 1):
            if sentence[i:i + word_len] == word:
                matches.append((i, i + word_len))
                distances.append(abs(i - word_start))

    if word_start >= 0:
        try:
            min_distance = min(distances)
            matches = [matches[i] for i in range(len(distances)) if distances[i] == min_distance]
        except:
            print(distances)

    all_token_ids = []
    for span in matches:
        token_ids = []
        all_token_ids.append(token_ids)
        for i in range(offset_mapping.shape[0]):
            if (span[0] <= offset_mapping[i, 1] - 1 and offset_mapping[i, 0] <= span[1] - 1):
                token_ids.append(i)

    return all_token_ids


def _get_lists_overlap(lists):
    def item_in_all(item, from_idx, lists):
        for j in range(from_idx, len(lists)):
            if item not in lists[j]:
                return False

        return True

    overlap = []
    for i in range(len(lists) - 1):
        for item in lists[i]:
            if item not in overlap and item_in_all(item, i + 1, lists):
                overlap.append(item)

    return list(overlap)


def analyse_contrapro(data, tokenize_fn, dir=None, file_name='data_analysis.json',
                      filter_by_word_ids=True, use_byte_string=False, force_reanalyse=False):
    data_analysis = None
    file_path = None
    if dir is not None:
        file_path = os.path.join(dir, file_name)
        if not force_reanalyse:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    data_analysis = json.load(f)
                    print(f'Loaded data analysis from the file: "{file_path}"')

    if data_analysis is None:
        print(f'Analyzing data...')

        data_analysis = []
        for data_point in tqdm.tqdm(data):
            d = data_point[-1]
            source = d['src segment']
            source_word_starts, source_words = _split_sentence(source, 'en')
            targets = [d['ref segment']] + [error['contrastive'] for error in d['errors']]
            target_options = [d['ref pronoun']] + [error['replacement'] for error in d['errors']]
            source_antecedent = d['src ante head']
            source_pronoun = d['src pronoun']
            source_tokenized = tokenize_fn(text=source, is_target=False, return_offsets_mapping=True)
            targets_tokenized = [tokenize_fn(text=target, is_target=True, return_offsets_mapping=True) for target in
                                 targets]

            if filter_by_word_ids:
                source_antecedent_word_id = d['src ante head id']
                source_antecedent_word_start = source_word_starts[source_antecedent_word_id - 1]
            else:
                source_antecedent_word_start = -1

            source_antecedent_token_ids = _find_token_ids(
                source_antecedent, source, source_tokenized['offset_mapping'][0],
                word_start=source_antecedent_word_start, use_byte_string=use_byte_string,
            )

            if len(source_antecedent_token_ids) != 1:
                print(source_antecedent_token_ids)
                print(source_antecedent)
                print(source)
                print()

            source_pronoun_token_ids = _find_token_ids(source_pronoun, source, source_tokenized['offset_mapping'][0],
                                                       use_byte_string=use_byte_string, )
            source_pronoun_token_ids = [token_ids for token_ids in source_pronoun_token_ids if
                                        token_ids != source_antecedent_token_ids]

            if 'ref ante head' in d and d['ref ante head'] is not None:
                target_antecedent = d['ref ante head']
                target_antecedent_token_ids = [
                    _find_token_ids(target_antecedent, targets[i], targets_tokenized[i]['offset_mapping'][0],
                                    use_byte_string=use_byte_string, )
                    for i in range(len(targets))
                ]
            else:
                target_antecedent_token_ids = None

            target_pronouns_token_ids = [
                _find_token_ids(target_options[i], targets[i], targets_tokenized[i]['offset_mapping'][0],
                                use_byte_string=use_byte_string, )
                for i in range(len(targets))
            ]

            if len(source_pronoun_token_ids) < 1:
                raise Exception()

            data_point_analysis = {}
            data_analysis.append(data_point_analysis)
            data_point_analysis['source_antecedent_token_ids'] = source_antecedent_token_ids
            data_point_analysis['source_pronoun_token_ids'] = source_pronoun_token_ids
            data_point_analysis['target_antecedent_token_ids'] = target_antecedent_token_ids
            data_point_analysis['target_pronouns_token_ids'] = target_pronouns_token_ids
            data_point_analysis['target_options'] = target_options

        if file_path is not None:
            with open(file_path, 'w') as f:
                print(f'Saving data analysis to the file: "{file_path}"...')
                json.dump(data_analysis, f)

    for data_point, analysis in zip(data, data_analysis):
        d = data_point[-1]
        if len(analysis['source_pronoun_token_ids']) < 1:
            raise Exception()

        d['source_antecedent_token_ids'] = analysis['source_antecedent_token_ids']
        d['source_pronoun_token_ids'] = analysis['source_pronoun_token_ids']
        d['target_antecedent_token_ids'] = analysis['target_antecedent_token_ids']
        d['target_pronouns_token_ids'] = analysis['target_pronouns_token_ids']
        d['target_options'] = analysis['target_options']

    return data_analysis

#
# def analyse_contrapro(
#         tokenized,
#         offset_mapping,
#         tokenize_fn,
#         filter_by_word_ids=True,
#         use_byte_string=False
# ):
#     d = data_point[-1]
#     source = d['src segment']
#     source_word_starts, source_words = _split_sentence(source, 'en')
#     targets = [d['ref segment']] + [error['contrastive'] for error in d['errors']]
#     target_options = [d['ref pronoun']] + [error['replacement'] for error in d['errors']]
#     source_antecedent = d['src ante head']
#     source_pronoun = d['src pronoun']
#     source_tokenized = tokenize_fn(text=source, is_target=False, return_offsets_mapping=True)
#     targets_tokenized = [tokenize_fn(text=target, is_target=True, return_offsets_mapping=True) for target in
#                          targets]
#
#     if filter_by_word_ids:
#         source_antecedent_word_id = d['src ante head id']
#         source_antecedent_word_start = source_word_starts[source_antecedent_word_id - 1]
#     else:
#         source_antecedent_word_start = -1
#
#     source_antecedent_token_ids = _find_token_ids(
#         source_antecedent, source, source_tokenized['offset_mapping'][0],
#         word_start=source_antecedent_word_start, use_byte_string=use_byte_string,
#     )
#
#     if len(source_antecedent_token_ids) != 1:
#         print(source_antecedent_token_ids)
#         print(source_antecedent)
#         print(source)
#         print()
#
#     source_pronoun_token_ids = _find_token_ids(source_pronoun, source, source_tokenized['offset_mapping'][0],
#                                                use_byte_string=use_byte_string, )
#     source_pronoun_token_ids = [token_ids for token_ids in source_pronoun_token_ids if
#                                 token_ids != source_antecedent_token_ids]
#
#     if 'ref ante head' in d and d['ref ante head'] is not None:
#         target_antecedent = d['ref ante head']
#         target_antecedent_token_ids = [
#             _find_token_ids(target_antecedent, targets[i], targets_tokenized[i]['offset_mapping'][0],
#                             use_byte_string=use_byte_string, )
#             for i in range(len(targets))
#         ]
#     else:
#         target_antecedent_token_ids = None
#
#     target_pronouns_token_ids = [
#         _find_token_ids(target_options[i], targets[i], targets_tokenized[i]['offset_mapping'][0],
#                         use_byte_string=use_byte_string, )
#         for i in range(len(targets))
#     ]
#
#     if len(source_pronoun_token_ids) < 1:
#         raise Exception()
#
#     data_point_analysis = {}
#     data_analysis.append(data_point_analysis)
#     data_point_analysis['source_antecedent_token_ids'] = source_antecedent_token_ids
#     data_point_analysis['source_pronoun_token_ids'] = source_pronoun_token_ids
#     data_point_analysis['target_antecedent_token_ids'] = target_antecedent_token_ids
#     data_point_analysis['target_pronouns_token_ids'] = target_pronouns_token_ids
#     data_point_analysis['target_options'] = target_options
