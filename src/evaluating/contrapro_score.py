import json
import math
import os
import pickle
import re
import warnings
from typing import List

import evaluate
import numpy as np
import torch
import tqdm

import visualization.plot_utils as plot_utils
from data.contrapro import load_contrapro, load_contrapro_with_context, analyse_contrapro, DataPoint
from visualization.attentions import plot_attentions


def select_sublist(base_list, include):
    return [base_list[i] for i, inc in enumerate(include) if inc]


def compute_metrics(metric, predictions, references):
    result = metric.compute(predictions=predictions, references=references)
    result = {"bleu": result["score"]}
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def save_results(model_name,
                 correct_total, total,
                 bleu,
                 results_dir, results_file):
    results_file = os.path.join(results_dir, results_file)
    print(f'Saving results to file: {results_file}...')
    with open(results_file, 'w') as f:
        f.write(f'model: {model_name}\n')
        f.write(f'correct {correct_total}\n')
        f.write(f'total {total}\n')
        f.write(f'accuracy {correct_total / total}\n')
        print(f'Accuracy {correct_total / total}')

        if bleu is not None:
            f.write(f'BLEU {round(bleu, 4)} ({bleu})\n')
            print(f'BLEU {round(bleu, 4)}')


def save_predictions(correct, selected, total_logprobs, selected_logprobs,
                     data: List[DataPoint], predictions, context_size,
                     results_dir, results_file, ):
    results_file = os.path.join(results_dir, results_file)  # 'results.txt'
    print(f'Saving results to file: {results_file}...')
    with open(results_file, 'w') as f:
        translations_generated = predictions is not None

        if predictions is None:
            predictions = [None for _ in data]

        print('Writing wrong predictions...')

        f.write('Wrong:\n')
        f.write('S - source\n')
        if context_size is not None and context_size > 0:
            f.write('SC - source context\n')
            f.write('TC - target context\n')
        f.write('R - reference\n')
        f.write('P - predicted\n')
        if translations_generated:
            f.write('G - generated\n')
        f.write('\n\n')

        i = -1
        pbar = tqdm.tqdm(zip(data, correct, selected, total_logprobs, selected_logprobs, predictions),
                         dynamic_ncols=True, ncols=200)
        d: DataPoint
        for d, c, sel, tp, sp, generated in pbar:
            s = d.source
            ts = d.targets
            sc = d.source_context
            tc = d.target_context

            i += 1
            if c:
                continue

            # generated = generate_fn(src=s, src_context=sc, tgt_context=tc)
            f.write(f'\tId: {i}\n')
            f.write(f'\tS: {s}\n')

            if context_size is not None and context_size > 0:
                f.write(f'\tSC: {sc}\n')
                f.write(f'\tTC: {tc}\n')

            f.write(f'\tR: {ts[0]} ({tp[0]})\n')
            f.write(f'\tP: {ts[sel]} ({sp}, {tp})\n')
            if translations_generated:
                f.write(f'\tG: {generated}\n')
            f.write('\n\n')

        print('Writing correct predictions...')

        f.write('\n\n\nCorrect:\n')
        f.write('S - source\n')
        if context_size is not None and context_size > 0:
            f.write('SC - context\n')
        f.write('P - predicted\n')
        if translations_generated:
            f.write('G - generated\n')
        f.write('\n\n')

        i = -1
        pbar = tqdm.tqdm(zip(data, correct, selected, total_logprobs, selected_logprobs, predictions),
                         dynamic_ncols=True, ncols=200)
        for d, c, sel, tp, sp, generated in pbar:
            s = d.source
            ts = d.targets
            sc = d.source_context
            tc = d.target_context

            i += 1
            if not c:
                continue

            # generated = generate_fn(src=s, src_context=sc, tgt_context=tc)
            f.write(f'\tId: {i}\n')
            f.write(f'\tS: {s}\n')

            if context_size is not None and context_size > 0:
                f.write(f'\tSC: {sc}\n')
                f.write(f'\tTC: {tc}\n')

            f.write(f'\tP: {ts[sel]} ({sp}, {tp})\n')
            if translations_generated:
                f.write(f'\tG: {generated}\n')
            f.write('\n\n')


def translate_data(data: List[DataPoint], generate_fn):
    print('Generating translations...')
    predictions = []
    pbar = tqdm.tqdm(data, dynamic_ncols=True, ncols=200)
    d: DataPoint
    for d in pbar:
        # s = d.source
        # sc = d.source_context
        #
        # (s, ts) = d[:2]
        # if len(d) > 3:
        #     sc = d[2]
        #     tc = d[3]
        # else:
        #     sc = None
        #     tc = None
        # json_data = d[-1]

        generated = generate_fn(data=d)
        predictions.append(generated)

    return predictions


def calculate_bleu(data: List[DataPoint], predictions):
    references = []
    d: DataPoint
    for d in data:
        ts = d.targets
        references.append(ts[0])

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=predictions, references=references)
    bleu = result["score"]
    print(f'BLEU {round(bleu, 4)} ({bleu})')
    return bleu


def get_results_dir(base_dir, model_name, results_dir_prefix=None, context_size=0, create=True,
                    use_name_slice_index=-1):
    results_dir_name = model_name.split("/")[use_name_slice_index]
    if results_dir_prefix is not None:
        results_dir_name = results_dir_prefix + results_dir_name
    # results_dir = f'../results/contrapro_pretrained/{results_dir_name}'
    results_dir = f'{base_dir}/{results_dir_name}'
    if context_size is not None and context_size > 0:
        results_dir += f'_ctx-{context_size}'

    if create:
        os.makedirs(results_dir, exist_ok=True)

    return results_dir


def score(data: List[DataPoint], score_contrastive_fn, save_attentions=True, limit_attentions=None):
    total = len(data)

    correct = []
    selected = []
    correct_total = 0
    ref_logprobs = []
    selected_logprobs = []
    tokens = []
    wrong_idxs = []
    all_target_encoded_ids = []
    all_target_logits = []
    pronoun_target_logits = []
    target_antecedents_specified = []
    all_total_logprobs = []
    self_attentions = []
    cross_attentions = []
    decoder_attentions = []
    pronoun_self_attentions = []
    pronoun_cross_attentions = []
    pronoun_decoder_attentions = []

    # src_phrase_indices,
    # src_context_phrase_indices,
    # tgt_phrases_indices,
    # tgt_context_phrases_indices,

    num_nans = 0
    i = 0
    pbar = tqdm.tqdm(data, dynamic_ncols=True, ncols=200)
    d: DataPoint
    for d in pbar:
        (
            total_logprobs,
            total_probs,
            toks,
            target_encoded_ids,
            target_logits,
            src_phrase_indices,
            src_context_phrase_indices,
            tgt_phrases_indices,
            tgt_context_phrases_indices,
            self_attention,
            cross_attention,
            decoder_attention,
        ) = score_contrastive_fn(data=d, return_attentions=True)

        d.src_phrase_indices = src_phrase_indices
        d.src_context_phrase_indices = src_context_phrase_indices
        d.tgt_phrases_indices = tgt_phrases_indices
        d.tgt_context_phrases_indices = tgt_context_phrases_indices

        tokens.append(toks)
        all_target_encoded_ids.append(target_encoded_ids)
        all_target_logits.append(target_logits)

        best_logprob = torch.argmax(torch.tensor(total_logprobs), dim=0)
        best_prob = torch.argmax(torch.tensor(total_probs), dim=0)

        any_nan = torch.tensor(total_logprobs).isnan().any()
        is_correct = best_logprob == 0 and not any_nan
        if any_nan:
            if num_nans < 10:
                warnings.warn(f'Nan logprobs for {i} {d.source} {d.targets[0]} {torch.tensor(total_logprobs)}')
            num_nans += 1

        selected.append(best_logprob if not any_nan else -1)
        correct.append(is_correct)
        ref_logprobs.append(total_logprobs[0])
        selected_logprobs.append(total_logprobs[best_logprob])
        all_total_logprobs.append([tlp.item() for tlp in total_logprobs])
        target_antecedents_specified.append(len(tgt_context_phrases_indices[0]) > 0)
        if is_correct:
            correct_total += 1
        else:
            wrong_idxs.append(i)

        if save_attentions:
            if limit_attentions is not None and limit_attentions > len(self_attentions):
                self_attentions.append(self_attention)
                cross_attentions.append(cross_attention)
                decoder_attentions.append(decoder_attention)

            pronoun_self_attentions.append([layer_attention[:, :, src_phrase_indices, :]
                                            for layer_attention in self_attention])
            pronoun_cross_attentions.append([[
                layer_attention[:, :, tgt_phrases_indices[j], :] for layer_attention in target_attention
            ] for j, target_attention in enumerate(cross_attention)])
            pronoun_decoder_attentions.append([[
                layer_attention[:, :, tgt_phrases_indices[j], :] for layer_attention in target_attention
            ] for j, target_attention in enumerate(decoder_attention)])
            pronoun_target_logits.append([target_ls[:, tgt_phrases_indices[j], :]
                                          for j, target_ls in enumerate(target_logits)])

            if len(pronoun_cross_attentions[i][0][0].shape) == 4:
                print('')
                print('len(pronoun_cross_attentions[i][0][0].shape) == 4')
                print('d.source', d.source)
                print('toks[1][0]', toks[1][0])
                print('d.targets[0]', d.targets[0])
                print('d.target_phrase', d.target_phrase)
                print('tgt_phrases_indices[0]', tgt_phrases_indices[0])
                print('pronoun_cross_attentions[i][0][0].shape', pronoun_cross_attentions[i][0][0].shape)
                print('')

            if len(src_phrase_indices) == 0 or len(tgt_phrases_indices[0]) == 0:
                print('')
                print('len(src_phrase_indices) == 0 or len(tgt_phrases_indices) == 0')
                print('d.source', d.source)
                print('toks[0]', toks[0])
                print('toks[1][0]', toks[1][0])
                print('d.targets[0]', d.targets[0])
                print('d.source_phrase', d.source_phrase)
                print('src_phrase_indices', src_phrase_indices)
                print('tgt_phrases_indices[0]', tgt_phrases_indices[0])
                print('')

        i += 1

    if num_nans > 0:
        warnings.warn(f'NaNs detected! Number of examples containing NaNs: {num_nans}')

    return (
        correct,
        selected,
        correct_total,
        ref_logprobs,
        selected_logprobs,
        total,
        self_attentions,
        cross_attentions,
        decoder_attentions,
        pronoun_self_attentions,
        pronoun_cross_attentions,
        pronoun_decoder_attentions,
        tokens,
        wrong_idxs,
        all_target_encoded_ids,
        all_target_logits,
        pronoun_target_logits,
        target_antecedents_specified,
        all_total_logprobs,
    )


# import memory_profiler


# @memory_profiler.profile
def score_and_plot_contrapro(score_contrastive_fn, generate_fn,
                             model_name,
                             results_dir, dataset_dir,
                             dataset_context_size=None,
                             filter_context_size=False,
                             use_json_lines=True,
                             limit_ids=None, limit_size=None, limit_plots=100,
                             results_suffix=None,
                             plot_separate_attentions=False,
                             plot_separate_heads=False,
                             generate_translations=True,
                             save_results_to_file=True,
                             save_detailed_results=True,
                             save_attentions=True,
                             batch_size=None):
    os.makedirs(results_dir, exist_ok=True)
    # results_dir = get_results_dir(base_dir, model_name, results_dir_prefix, dataset_context_size,
    #                               create=True, use_name_slice_index=use_name_slice_index)

    data = load_contrapro_with_context(dataset_dir, dataset_context_size,
                                       filter_context_size=filter_context_size,
                                       limit_ids=limit_ids, use_json_lines=use_json_lines)

    # if dataset_context_size is not None and dataset_context_size > 0:
    #     data = load_contrapro_with_context(dataset_dir, dataset_context_size,
    #                                        filter_context_size=filter_context_size,
    #                                        limit_ids=limit_ids)
    # else:
    #     data = load_contrapro(dataset_dir,
    #                           filter_context_size=filter_context_size,
    #                           limit_ids=limit_ids)

    if limit_size is not None and limit_size > 0:
        data = data[:min(limit_size, len(data))]

    # data_analysis = analyse_contrapro(data, tokenize_fn, results_dir,
    #                                   force_reanalyse=force_reanalyse,
    #                                   use_byte_string=use_byte_string_for_data_analysis)

    # if batch_size is not None and batch_size > 1:
    #     (
    #         correct,
    #         selected,
    #         correct_total,
    #         ref_logprobs,
    #         selected_logprobs,
    #         total,
    #         self_attentions,
    #         cross_attentions,
    #         decoder_attentions,
    #         pronoun_self_attentions,
    #         pronoun_cross_attentions,
    #         pronoun_decoder_attentions,
    #         tokens,
    #         wrong_idxs,
    #         all_target_encoded_ids,
    #         all_target_logits,
    #         pronoun_target_logits,
    #         target_antecedents_specified,
    #         all_total_logprobs,
    #     ) = score_batched(data, score_contrastive_fn,
    #                       batch_size=batch_size,
    #                       save_attentions=save_attentions or limit_plots > 0)
    # else:

    num_batches = math.ceil(len(data) / batch_size) if batch_size is not None else 1
    if num_batches > 1:
        print(f'Processing {len(data)} examples in {num_batches} batches...')

    all_correct_total = 0
    all_total = 0
    all_correct = []
    all_selected = []
    all_total_logprobs = []
    all_selected_logprobs = []
    all_predictions = []
    all_bleus = []
    for batch_id in range(num_batches):
        if num_batches > 1:
            print(f'Batch {batch_id + 1}/{num_batches}...')
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, len(data))
            data_batch = data[start:end]
        else:
            data_batch = data

        (
            correct,
            selected,
            correct_total,
            ref_logprobs,
            selected_logprobs,
            total,
            self_attentions,
            cross_attentions,
            decoder_attentions,
            pronoun_self_attentions,
            pronoun_cross_attentions,
            pronoun_decoder_attentions,
            tokens,
            wrong_idxs,
            all_target_encoded_ids,
            all_target_logits,
            pronoun_target_logits,
            target_antecedents_specified,
            total_logprobs,
        ) = score(data_batch, score_contrastive_fn,
                  save_attentions=save_attentions or limit_plots > 0,
                  limit_attentions=limit_plots)

        all_correct_total += correct_total
        all_total += total
        all_correct.extend(correct)
        all_selected.extend(selected)
        all_total_logprobs.extend(total_logprobs)
        all_selected_logprobs.extend(selected_logprobs)

        if generate_translations:
            predictions = translate_data(data_batch, generate_fn)
            bleu = calculate_bleu(data_batch, predictions)
            all_predictions.extend(predictions)
            all_bleus.append(bleu)
        else:
            predictions = None
            bleu = None

        print(f'Partial accuracy: {correct_total / total}')

        if save_results_to_file and save_attentions:
            results = {
                'tokens': tokens,
                'correct': [torch.tensor(c) for c in correct],
                'total': total,
                # 'target_encoded_ids': all_target_encoded_ids,
                'phrase_target_logits': pronoun_target_logits,
                'phrase_attentions': {
                    'self_attentions': pronoun_self_attentions,
                    'cross_attentions': pronoun_cross_attentions,
                    'decoder_attentions': pronoun_decoder_attentions,
                },
                'data': data_batch,
                'accuracy': correct_total / total,
                'bleu': bleu,
            }
            part_str = f'.part-{batch_id + 1}' if num_batches > 1 else ''
            if results_suffix is not None:
                results_file = f'results_{results_suffix}{part_str}.pkl'
            else:
                results_file = f'results{part_str}.pkl'
            # results_file = f'results_{results_suffix}{part_str}.pkl' if results_suffix is not None else f'results{part_str}.pkl'
            with open(os.path.join(results_dir, results_file), 'wb') as f:
                pickle.dump(results, f)

        if batch_size is not None:
            batch_limit_plots = max(limit_plots - (batch_id * batch_size), 0) if limit_plots is not None else None
        else:
            batch_limit_plots = limit_plots

        plot_attentions(results_dir, correct,
                        self_attentions, cross_attentions, decoder_attentions, tokens,
                        batch_limit_plots, plot_separate_attentions, plot_separate_heads, results_suffix)

        del tokens
        del self_attentions, cross_attentions, decoder_attentions
        del pronoun_self_attentions, pronoun_cross_attentions, pronoun_decoder_attentions

    if num_batches > 1 and generate_translations:
        all_bleu = calculate_bleu(data, all_predictions)
    else:
        all_bleu = bleu

    if save_results_to_file:
        results_file = f'results_{results_suffix}.txt' \
            if results_suffix is not None else 'results.txt'
        save_results(model_name,
                     all_correct_total, all_total,
                     all_bleu,
                     results_dir, results_file)
        if save_detailed_results:
            predictions_results_file = f'results_preds_{results_suffix}.txt' \
                if results_suffix is not None else 'results_preds.txt'
            save_predictions(all_correct, all_selected, all_total_logprobs, all_selected_logprobs,
                             data, all_predictions, dataset_context_size,
                             results_dir, predictions_results_file)

    return {
        'correct': all_correct,
        'total': all_total,
        'data': data,
        'accuracy': all_correct_total / all_total,
        'bleu': all_bleu,
    }
