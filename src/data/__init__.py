import os
from typing import List, Union, Iterable

import datasets

# from data.contextual_dataset import ContextualDataset
from data.contextual_iwslt2017 import ContextualIWSLT2017
from data.contextual_iwslt2017 import PAIRS as _IWSLT2017_LANGUAGE_PAIRS
from data.contrapro_training_dataset import ContraPro


# SPACY_LANGUAGE_MODELS = {
#     'de': 'de_core_news_sm',
#     'el': 'el_core_news_sm',
#     'en': 'en_core_web_sm',
#     'es': 'es_core_news_sm',
#     'it': 'it_core_news_sm',
#     'nl': 'nl_core_news_sm',
# }


def concatenate_current_and_context(current: Union[str, List[str]],
                                    context: Union[List[str], List[List[str]]],
                                    sep_token):
    batched = isinstance(current, list)
    if not batched:
        assert isinstance(current, str) and isinstance(context, list), \
            f'current: {isinstance(current, str)}, context: {isinstance(context, list)}'
        current = [current]
        context = [context]
    else:
        assert isinstance(context, list)
        assert len(current) == len(context) and all(isinstance(c, list) for c in context), \
            print(f'Lengths of the current and context lists do not match: {len(current)} vs {len(context)}')

    concatenated = []
    for c, cc in zip(current, context):
        if cc is None or len(cc) == 0:
            concatenated.append(c)
        else:
            if sep_token is not None:
                concatenated.append(f'{sep_token} '.join(cc + [c]))
            else:
                concatenated.append(' '.join(cc + [c]))

    if batched:
        return concatenated
    else:
        return concatenated[0]


def _process_dataset(
        dataset,
        tokenized_dataset,
        tokenizer,
        src_lang, tgt_lang,
        context_sep_token,
        src_ctx_size=1,
        tgt_ctx_size=1,
        max_length=200,
        split_dataset=True,
        valid_size=0.1,
        test_size=0.1,
        train_limit_size=None,
        valid_limit_size=None,
        test_limit_size=None,
        train_limit_start_index=0,
        valid_limit_start_index=0,
        test_limit_start_index=0,
        tokenizer_language_dict=None,
        set_tokenizer_languages=True,
        include_forced_bos_token=True,
        seed=42
):
    if tokenized_dataset is None:
        tokenizer_src_lang = src_lang
        tokenizer_tgt_lang = tgt_lang
        if tokenizer_language_dict is not None:
            tokenizer_src_lang = tokenizer_language_dict[tokenizer_src_lang]
            tokenizer_tgt_lang = tokenizer_language_dict[tokenizer_tgt_lang]

        if split_dataset:
            valid_test_size = valid_size + test_size
            test_to_valid_and_test_ratio = test_size / valid_test_size
            dataset = dataset.train_test_split(test_size=valid_test_size, seed=seed)
            test_valid_dataset = dataset['test'].train_test_split(test_size=test_to_valid_and_test_ratio, seed=seed)
            dataset['valid'] = test_valid_dataset['train']
            dataset['test'] = test_valid_dataset['test']

        # tgt_nlp_model = spacy.load(SPACY_LANGUAGE_MODELS[tgt_lang])
        # ignore_poss = ('NUM', 'PUNCT', 'SYM', 'X', 'EOL', 'SPACE')
        # rng = np.random.default_rng()
        sep_token = context_sep_token

        def tokenize_with_context(examples):
            if set_tokenizer_languages:
                tokenizer.src_lang = tokenizer_src_lang
                tokenizer.tgt_lang = tokenizer_tgt_lang
            sources = []
            targets = []
            for t, c in zip(examples['translation'], examples['context']):
                src_ctx = [sc for sc in c[src_lang][-src_ctx_size:] if len(sc) > 0]
                # src_concatenated = f' {sep_token} '.join(src_ctx + [t[src_lang]])
                src_concatenated = concatenate_current_and_context(t[src_lang], src_ctx, sep_token)
                sources.append(src_concatenated)

                tgt_ctx = [tc for tc in c[tgt_lang][-tgt_ctx_size:] if len(tc) > 0]
                # tgt_concatenated = f' {sep_token} '.join(tgt_ctx + [t[tgt_lang]])
                tgt_concatenated = concatenate_current_and_context(t[tgt_lang], tgt_ctx, sep_token)
                targets.append(tgt_concatenated)

            tokenized = tokenizer(sources, text_target=targets, max_length=max_length, truncation=True)
            if include_forced_bos_token:
                tokenized['forced_bos_token_id'] = [tokenizer.lang_code_to_id[tokenizer_tgt_lang]] * len(targets)
            return tokenized

        def tokenize_without_context(examples):
            if set_tokenizer_languages:
                tokenizer.src_lang = tokenizer_src_lang
                tokenizer.tgt_lang = tokenizer_tgt_lang
            sources = [t[src_lang] for t in examples['translation']]
            targets = [t[tgt_lang] for t in examples['translation']]

            tokenized = tokenizer(sources, text_target=targets, max_length=max_length, truncation=True)
            if include_forced_bos_token:
                tokenized['forced_bos_token_id'] = [tokenizer.lang_code_to_id[tokenizer_tgt_lang]] * len(targets)
            return tokenized

        print('Tokenized dataset not found in the specified location.\n'
              'Processing the dataset (it will take a while)...')
        tokenize_fn = tokenize_with_context if src_ctx_size > 0 else tokenize_without_context
        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)

    train_dataset = tokenized_dataset['train'].select(range(train_limit_start_index, train_limit_size)) \
        if train_limit_size is not None else tokenized_dataset['train']
    eval_dataset = tokenized_dataset['valid'].select(range(valid_limit_start_index, valid_limit_size)) \
        if valid_limit_size is not None else tokenized_dataset['valid']
    test_dataset = tokenized_dataset['test'].select(range(test_limit_start_index, test_limit_size)) \
        if test_limit_size is not None else tokenized_dataset['test']

    print('train_dataset')
    print(train_dataset)

    for i in range(5):
        ids = train_dataset[i]['input_ids']
        labels = train_dataset[i]['labels']
        print(tokenizer.convert_ids_to_tokens(ids))
        print(tokenizer.convert_ids_to_tokens(labels))

    print(f'Train dataset: {len(train_dataset)}')
    print(f'Eval dataset: {len(eval_dataset)}')
    print(f'Test dataset: {len(test_dataset)}')

    return train_dataset, eval_dataset, test_dataset, tokenized_dataset



def load_iwslt2017_dataset_raw(base_data_dir,
                               src_lang, tgt_lang,
                               src_ctx_size=1,
                               tgt_ctx_size=1, ):
    if f'{src_lang}-{tgt_lang}' in _IWSLT2017_LANGUAGE_PAIRS:
        lang1, lang2 = src_lang, tgt_lang
        lang1_ctx_size, lang2_ctx_size = src_ctx_size, tgt_ctx_size
    elif f'{src_lang}-{tgt_lang}' in _IWSLT2017_LANGUAGE_PAIRS:
        lang1, lang2 = tgt_lang, src_lang
        lang1_ctx_size, lang2_ctx_size = tgt_ctx_size, src_ctx_size
    else:
        raise AttributeError(
            f'Language pair {src_lang}-{tgt_lang} not available. Choose from {_IWSLT2017_LANGUAGE_PAIRS}.')

    max_ctx_size = max(src_ctx_size, tgt_ctx_size)
    data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', f'ctx{max_ctx_size}')
    ds_builder = ContextualIWSLT2017('__main__', f'{lang1}-{lang2}-ctx{max_ctx_size}',
                                     pair=f'{src_lang}-{tgt_lang}', is_multilingual=False,
                                     lang1_ctx_size=lang1_ctx_size,
                                     lang2_ctx_size=lang2_ctx_size)
    ds_builder.download_and_prepare(data_dir)
    dataset = ds_builder.as_dataset()

    dataset['valid'] = dataset['validation']
    del dataset['validation']

    print(f'Dataset ({lang1}-{lang2}, ctx{max_ctx_size}) loaded '
          f'({len(dataset["train"])} / {len(dataset["valid"])} / {len(dataset["test"])} elements)')
    for i in range(5):
        print(dataset['train'][i])

    return dataset


def load_iwslt2017_dataset(tokenizer, base_data_dir,
                           src_lang, tgt_lang,
                           context_sep_token,
                           src_ctx_size=1,
                           tgt_ctx_size=1,
                           valid_size=0.1,
                           test_size=0.1,
                           max_length=200,
                           split_dataset=False,
                           tokenizer_language_dict=None,
                           set_tokenizer_languages=False,
                           include_forced_bos_token=False,
                           model_name='m2m',
                           seed=42):
    if f'{src_lang}-{tgt_lang}' in _IWSLT2017_LANGUAGE_PAIRS:
        lang1, lang2 = src_lang, tgt_lang
        lang1_ctx_size, lang2_ctx_size = src_ctx_size, tgt_ctx_size
    elif f'{src_lang}-{tgt_lang}' in _IWSLT2017_LANGUAGE_PAIRS:
        lang1, lang2 = tgt_lang, src_lang
        lang1_ctx_size, lang2_ctx_size = tgt_ctx_size, src_ctx_size
    else:
        raise AttributeError(
            f'Language pair {src_lang}-{tgt_lang} not available. Choose from {_IWSLT2017_LANGUAGE_PAIRS}.')

    max_ctx_size = max(src_ctx_size, tgt_ctx_size)

    name_suffix = f'{src_lang}-{tgt_lang}_{src_ctx_size}-{tgt_ctx_size}_{model_name}'
    if context_sep_token is None:
        name_suffix += '_no-sep-token'

    processed_file_name = f'iwslt2017_{name_suffix}_processed.hf'
    processed_data_path = os.path.join(base_data_dir, processed_file_name)
    try:
        tokenized_dataset = datasets.load_from_disk(processed_data_path)
        dataset = None
        loaded_from_disk = True
        print(f'Loaded tokenized dataset ({src_lang}-{tgt_lang}, ctx-{max_ctx_size}) '
              f'from the specified location {processed_data_path}.')
    except FileNotFoundError:
        print(f'Tokenized dataset not found in the specified location ({lang1}-{lang2}, ctx-{max_ctx_size}).\n'
              'Processing the dataset (it will take a while)...')
        tokenized_dataset = None
        loaded_from_disk = False

        dataset = load_iwslt2017_dataset_raw(base_data_dir, src_lang, tgt_lang, src_ctx_size, tgt_ctx_size)

    (
        train_dataset, eval_dataset, test_dataset, tokenized_dataset
    ) = _process_dataset(
        dataset,
        tokenized_dataset,
        tokenizer,
        context_sep_token=context_sep_token,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_ctx_size=src_ctx_size,
        tgt_ctx_size=tgt_ctx_size,
        valid_size=valid_size,
        test_size=test_size,
        max_length=max_length,
        split_dataset=split_dataset,
        tokenizer_language_dict=tokenizer_language_dict,
        seed=seed,
        set_tokenizer_languages=set_tokenizer_languages,
        include_forced_bos_token=include_forced_bos_token,
    )

    if not loaded_from_disk:
        tokenized_dataset.save_to_disk(processed_data_path)

    del tokenized_dataset
    return train_dataset, eval_dataset, test_dataset, dataset


def load_contrapro_dataset_raw(dataset_dir, processed_dataset_dir,
                               source_context_size, target_context_size,
                               test_size, split_seed):
    print(f'Loading ContraPro dataset from {dataset_dir}...')
    max_context_size = max(source_context_size, target_context_size)
    full_dataset_dir = os.path.join(dataset_dir, f'ctx{max_context_size}')
    ds_builder = ContraPro(full_dataset_dir,
                           base_path=dataset_dir,
                           ctx_size=max_context_size,
                           files_base_name='contrapro')
    ds_builder.download_and_prepare(processed_dataset_dir)
    ds = ds_builder.as_dataset('train')

    if test_size is not None and test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=split_seed)

    return ds

