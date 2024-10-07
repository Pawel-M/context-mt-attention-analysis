import argparse
import os
import shutil
from functools import partial

import datasets
import evaluate
import numpy as np
import torch
import torchinfo
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer

from modeling.opus_mt_adjustable import AdjustableMarianMTModel
from evaluating.common_tokenization import tokenize_with_context, tokenize_all_with_context
# from modeling.opus_mt_tokenization import tokenize_with_context, tokenize_all_with_context
from evaluating.opus_mt_functions import generate_translation_raw


def load_iwslt2017(base_data_dir,
                   src_lang, tgt_lang,
                   src_ctx_size, tgt_ctx_size,
                   sample_ctx_size, match_ctx_size, ):
    from data import load_iwslt2017_dataset_raw

    if not sample_ctx_size:
        raw_dataset = load_iwslt2017_dataset_raw(base_data_dir,
                                                 src_lang, tgt_lang,
                                                 src_ctx_size=src_ctx_size,
                                                 tgt_ctx_size=tgt_ctx_size, )
    else:
        context_sizes = []
        if match_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(s, t) for s in range(src_ctx_size + 1) for t in range(tgt_ctx_size + 1)]

        print(f'Loading datasets with context sizes: {context_sizes}')

        raw_datasets = []
        for src_cs, tgt_cs in context_sizes:
            raw_dataset = load_iwslt2017_dataset_raw(base_data_dir,
                                                     src_lang, tgt_lang,
                                                     src_ctx_size=src_cs,
                                                     tgt_ctx_size=tgt_cs, )
            raw_datasets.append(raw_dataset)
        raw_dataset = datasets.interleave_datasets(raw_datasets)

    return raw_dataset


def load_contrapro(dataset_dir, processed_dataset_dir,
                   src_ctx_size, tgt_ctx_size):
    from data import load_contrapro_dataset_raw
    raw_dataset = load_contrapro_dataset_raw(dataset_dir, processed_dataset_dir,
                                             src_ctx_size, tgt_ctx_size,
                                             0.0, 1)

    return raw_dataset


def load_dataset(dataset_name, base_data_dir, raw_data_dir,
                 model_name,
                 src_lang, tgt_lang,
                 src_ctx_size, tgt_ctx_size,):
    if dataset_name == 'iwslt2017':
        dataset = load_iwslt2017(
            base_data_dir,
            src_lang, tgt_lang,
            src_ctx_size, tgt_ctx_size,
            False, True,
        )
    elif dataset_name == 'ContraPro':
        dataset_dir = raw_data_dir
        processed_dataset_dir = os.path.join(base_data_dir, f'{model_name}_ctx_{src_ctx_size}')
        dataset = load_contrapro(dataset_dir, processed_dataset_dir,
                                 src_ctx_size, tgt_ctx_size,)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported!')

    return dataset


def translate(examples, model, tokenizer, src_lang, tgt_lang, src_ctx_size, tgt_ctx_size, beam_size, max_length):
    sources = [t[src_lang] for t in examples['translation']]
    targets = [t[tgt_lang] for t in examples['translation']]
    sources_context = [t[src_lang] for t in examples['context']]
    targets_context = [t[tgt_lang] for t in examples['context']]
    generated = generate_translation_raw(model, tokenizer, model.device,
                                         src_ctx_size, tgt_ctx_size,
                                         sources, sources_context, targets_context,
                                         num_beams=beam_size, max_len=max_length)

    examples['generated'] = generated
    examples['target'] = targets
    return examples


def load_dataset_and_translate(model, tokenizer, model_name,
                               dataset_name,
                               base_data_dir, raw_data_dir,
                               src_lang, tgt_lang,
                               src_ctx_size, tgt_ctx_size,
                               beam_size, max_length,
                               results_dir,
                               dataset_splits=None,
                               results_suffix=None,
                               batch_size=12):
    dataset = load_dataset(dataset_name, base_data_dir, raw_data_dir,
                           model_name,
                           src_lang, tgt_lang,
                           src_ctx_size, tgt_ctx_size,)
    # if dataset_name == 'iwslt2017':
    #     dataset = load_iwslt2017(
    #         base_data_dir,
    #         src_lang, tgt_lang,
    #         src_ctx_size, tgt_ctx_size,
    #         sample_ctx_size, match_ctx_size,
    #     )
    # elif dataset_name == 'ContraPro':
    #     dataset_dir = raw_data_dir
    #     processed_dataset_dir = os.path.join(base_data_dir, f'{model_name}_ctx_{src_ctx_size}')
    #     dataset = load_contrapro(dataset_dir, processed_dataset_dir,
    #                              src_ctx_size, tgt_ctx_size,
    #                              args.test_size, args.split_seed)
    # else:
    #     raise ValueError(f'Dataset {dataset_name} is not supported!')

    print('dataset', dataset)

    if dataset_splits is None:
        dss = [(None, dataset)]
    else:
        dss = [(ds, dataset[ds]) for ds in dataset_splits]

    bleus = {}
    for ds_name, ds in dss:
        print(f'Translating {ds_name} split with {len(ds)} examples...')
        translate_fn = partial(translate, model=model, tokenizer=tokenizer,
                               src_lang=src_lang, tgt_lang=tgt_lang,
                               src_ctx_size=src_ctx_size, tgt_ctx_size=tgt_ctx_size,
                               beam_size=beam_size, max_length=max_length)
        ds = ds.map(translate_fn, batched=True, batch_size=batch_size, keep_in_memory=True)

        results_suffix = f'.{results_suffix}' if results_suffix is not None else ''

        ds_name_suffix = f'.{ds_name}' if ds_name is not None else ''

        generated = ds['generated']
        targets = ds['target']
        generated_file_name = f'{dataset_name}{ds_name_suffix}{results_suffix}.preds.txt'
        target_file_name = f'{dataset_name}{ds_name_suffix}{results_suffix}.targets.txt'

        print(f'Saving results to {os.path.join(results_dir, generated_file_name)}')
        with open(os.path.join(results_dir, generated_file_name), 'w') as f:
            f.write('\n'.join(generated))
        print(f'Saving targets to {os.path.join(results_dir, target_file_name)}')
        with open(os.path.join(results_dir, target_file_name), 'w') as f:
            f.write('\n'.join(targets))

        bleu_metric = sacrebleu.BLEU()
        bleu = bleu_metric.corpus_score(hypotheses=generated, references=[targets])
        print(f'BLEU: {bleu.score}')
        results_file_name = f'{dataset_name}{ds_name_suffix}{results_suffix}.results.txt'
        print(f'Saving results to {os.path.join(results_dir, results_file_name)}')
        with open(os.path.join(results_dir, results_file_name), 'w') as f:
            f.write(f'{bleu.score}\n')
            f.write(str(bleu))

        bleus[ds_name] = bleu.score

    return bleus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str, help='model name (from huggingface) or model path')
    parser.add_argument("--tokenizer-path", default=None, type=str,
                        help='model name (from huggingface) or model path (defaults to --model-path)')
    parser.add_argument("--model-name", required=True, type=str, help="model name for saving parsed dataset")
    parser.add_argument("--results-dir", default='.', type=str, help='directory to save the results')
    parser.add_argument("--results-suffix", default=None, type=str, help='suffix added to the results file')
    parser.add_argument("--dataset", default='iwslt2017', type=str, help='the dataset to translate')
    parser.add_argument("--dataset-splits", default=None, type=str, nargs='+',
                        help='the dataset splits to translate')
    parser.add_argument("--base-data-dir", default=None, type=str,
                        help='base directory to save loaded data')
    parser.add_argument("--raw-data-dir", default='.', type=str,
                        help='base directory to load raw data (eg. ContraPro or ctxpro dataset dir)')
    # parser.add_argument("--test-size", default=None, type=int, help='size of the test set')
    # parser.add_argument("--split-seed", default=1, type=int, help='seed for splitting the dataset')
    parser.add_argument("--src-lang", default='en', type=str, help='source language')
    parser.add_argument("--tgt-lang", default='de', type=str, help='target language')
    parser.add_argument("--src-ctx-size", default=0, type=int, help='size of the source side')
    parser.add_argument("--tgt-ctx-size", default=0, type=int, help='size of the target side')
    # parser.add_argument("--sample-ctx-size", default=False, action='store_true',
    #                     help='sample the size of the context')
    # parser.add_argument("--match-ctx-size", default=False, action='store_true',
    #                     help='match the sampled size of the source and target context')
    parser.add_argument("--beam-size", default=5, type=int, help='beam size for generating translations')
    parser.add_argument("--max-length", default=200, type=int, help='maximum length of the sentences')
    # parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--no-sep-token", action='store_true', default=False,
                        help="if set, doesn't add the separator token")

    args = parser.parse_args()

    dataset_name = args.dataset
    results_dir = args.results_dir
    model_name = args.model_name
    base_data_dir = args.base_data_dir
    raw_data_dir = args.raw_data_dir
    add_sep_token = not args.no_sep_token
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_ctx_size = args.src_ctx_size
    tgt_ctx_size = args.tgt_ctx_size
    # sample_ctx_size = args.sample_ctx_size
    # match_ctx_size = args.match_ctx_size
    beam_size = args.beam_size
    max_length = args.max_length

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    model = AdjustableMarianMTModel.from_pretrained(model_path)
    model = model.to(device)

    print('tokenizer', tokenizer)
    print('model', model)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    bleus = load_dataset_and_translate(model, tokenizer, model_name,
                                       dataset_name,
                                       base_data_dir, raw_data_dir,
                                       src_lang, tgt_lang,
                                       src_ctx_size, tgt_ctx_size,
                                       # sample_ctx_size, match_ctx_size,
                                       beam_size, max_length,
                                       results_dir,
                                       dataset_splits=args.dataset_splits,
                                       results_suffix=args.results_suffix,
                                       batch_size=1)
