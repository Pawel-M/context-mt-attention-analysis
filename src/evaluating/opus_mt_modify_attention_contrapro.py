import argparse
import functools

import torch
from transformers import AutoTokenizer

from evaluating.contrapro_score import score_and_plot_contrapro
from evaluating.common_functions import score_contrastive_modifying_attention
from evaluating.opus_mt_functions import generate_translation
from evaluating.utils import generate_full_heads_list, parse_heads_list
from modeling.opus_mt_adjustable import AdjustableMarianMTModel


# from modeling.opus_mt_tokenizer import OffsetEnabledMarianTokenizer


def score_modified_attention(model, tokenizer, model_name,
                             results_dir, dataset_dir,
                             modify_to_value: float,
                             encoder_layers_heads,
                             phrase_cross_attention_layer_heads,
                             context_cross_attention_layer_heads,
                             decoder_attention_layer_heads,
                             decoder_after_attention_layer_heads,
                             dataset_context_size=None, source_context_size=None, target_context_size=None,
                             filter_context_size=False, limit_ids=None,
                             limit_size=None, limit_plots=200,
                             results_suffix=None,
                             plot_separate_attentions=False, plot_separate_heads=False,
                             generate_translations=True, max_len=300, num_beams=5,
                             save_attentions=True, batch_size=None):
    source_context_size = 0 if source_context_size is None else source_context_size
    target_context_size = 0 if target_context_size is None else target_context_size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scorer = functools.partial(score_contrastive_modifying_attention, model=model,
                               source_context_size=source_context_size, target_context_size=target_context_size,
                               modify_to_value=modify_to_value,
                               encoder_layers_heads=encoder_layers_heads,
                               phrase_cross_attention_layer_heads=phrase_cross_attention_layer_heads,
                               context_cross_attention_layer_heads=context_cross_attention_layer_heads,
                               decoder_attention_layer_heads=decoder_attention_layer_heads,
                               decoder_after_attention_layer_heads=decoder_after_attention_layer_heads,
                               tokenizer=tokenizer, device=device)

    generator = functools.partial(generate_translation, model=model, tokenizer=tokenizer, device=device,
                                  source_context_size=source_context_size, target_context_size=target_context_size,
                                  max_len=max_len, num_beams=num_beams)

    results = score_and_plot_contrapro(
        model_name=model_name,
        score_contrastive_fn=scorer,
        generate_fn=generator,
        results_dir=results_dir,
        dataset_dir=dataset_dir,
        dataset_context_size=dataset_context_size,
        filter_context_size=filter_context_size,
        limit_ids=limit_ids,
        limit_size=limit_size,
        limit_plots=limit_plots,
        results_suffix=results_suffix,
        plot_separate_attentions=plot_separate_attentions,
        plot_separate_heads=plot_separate_heads,
        generate_translations=generate_translations,
        save_results_to_file=True,
        save_detailed_results=False,
        save_attentions=save_attentions,
        batch_size=batch_size,
    )
    return results


def get_attention_layer_heads(attention_type, layer, head):
    encoder_layers_heads = []
    phrase_cross_attention_layer_heads = []
    context_cross_attention_layer_heads = []
    decoder_attention_layer_heads = []
    decoder_after_attention_layer_heads = []
    if attention_type == 'encoder':
        encoder_layers_heads.append((layer, head))
    elif attention_type == 'phrase_cross':
        phrase_cross_attention_layer_heads.append((layer, head))
    elif attention_type == 'context_cross':
        context_cross_attention_layer_heads.append((layer, head))
    elif attention_type == 'decoder':
        decoder_attention_layer_heads.append((layer, head))
    elif attention_type == 'decoder_after':
        decoder_after_attention_layer_heads.append((layer, head))
    else:
        raise ValueError(f'Unknown attention type: {attention_type}')
    return (encoder_layers_heads,
            phrase_cross_attention_layer_heads,
            context_cross_attention_layer_heads,
            decoder_attention_layer_heads,
            decoder_after_attention_layer_heads)


def get_combined_attention_layer_heads(heads_list):
    all_encoder_layers_heads = []
    all_phrase_cross_attention_layer_heads = []
    all_context_cross_attention_layer_heads = []
    all_decoder_attention_layer_heads = []
    all_decoder_after_attention_layer_heads = []

    for attention_type, layer, head in heads_list:
        (encoder_layers_heads,
         phrase_cross_attention_layer_heads,
         context_cross_attention_layer_heads,
         decoder_attention_layer_heads,
         decoder_after_attention_layer_heads) = get_attention_layer_heads(attention_type, layer, head)
        all_encoder_layers_heads.extend(encoder_layers_heads)
        all_phrase_cross_attention_layer_heads.extend(phrase_cross_attention_layer_heads)
        all_context_cross_attention_layer_heads.extend(context_cross_attention_layer_heads)
        all_decoder_attention_layer_heads.extend(decoder_attention_layer_heads)
        all_decoder_after_attention_layer_heads.extend(decoder_after_attention_layer_heads)

    return (all_encoder_layers_heads,
            all_phrase_cross_attention_layer_heads,
            all_context_cross_attention_layer_heads,
            all_decoder_attention_layer_heads,
            all_decoder_after_attention_layer_heads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default='.', type=str, help='directory to save the results')
    parser.add_argument("--results-suffix", default=None, type=str, help='suffix added to the results file')
    parser.add_argument("--save-results-file", default=None, type=str,
                        help='file to save the results')
    parser.add_argument("--contrapro-dir", default='.', type=str,
                        help='directory with ContraPro dataset, expects ot have ctx1, ctx2, etc. folders inside')
    parser.add_argument("--contrapro-ctx-size", default=None, type=int, help='context size of the dataset')
    parser.add_argument("--filter-context-size", action='store_true', default=False,
                        help='if set, only includes the sentences where ante_distance is <= to `--contrapro-ctx-size`.')
    parser.add_argument("--src-lang", default='en', type=str, help='source language')
    parser.add_argument("--tgt-lang", default='de', type=str, help='target language')
    parser.add_argument("--src-ctx-size", default=0, type=int, help='context size of the source side')
    parser.add_argument("--tgt-ctx-size", default=0, type=int, help='context size of the target side')
    parser.add_argument("--model-path", required=True, type=str, help='model name (from huggingface) or model path')
    parser.add_argument("--tokenizer-path", default=None, type=str,
                        help='model name (from huggingface) or model path (defaults to --model-path)')
    parser.add_argument("--limit-dataset-size", default=None, type=int, help='limit the numer of examples')
    parser.add_argument("--limit-plots", default=None, type=int, help='limit the numer of attention plots')
    parser.add_argument("--generate-translations", action='store_true', default=False,
                        help='if set, generates translations from the source and context')
    parser.add_argument("--max-len", default=300, type=int, help='maximum length of the generated translations')
    parser.add_argument("--num-beams", default=5, type=int, help='number of beams in generating translations')
    parser.add_argument("--save-attention-scores", action='store_true', default=False,
                        help='if set, saves attention scores to a file')
    parser.add_argument("--modify-heads", default=None, type=str, nargs='+',
                        help='list of heads to modify in the form (attention_relation, layer, head), '
                             'where attention_relation is one of: '
                             '"encoder", "phrase_cross", "context_cross", "decoder", "decoder_after"')
    parser.add_argument("--modify-all-model-heads", action='store_true', default=False,
                        help='if set, disables all heads')
    parser.add_argument("--limit-layers", default=None, type=int, nargs='+',
                        help='list of layers to limit the modifications to')
    parser.add_argument("--limit-heads", default=None, type=int, nargs='+',
                        help='list of heads to limit the modifications to')
    parser.add_argument("--modify-heads-to-values", default=None, type=float, nargs='+',
                        help='list of values to set the attention scores to')
    parser.add_argument("--modify-heads-simultaneously", action='store_true', default=False,
                        help='if set, modifies all heads simultaneously')
    parser.add_argument("--batch-size", default=None, type=int, help='batch size for scoring')

    args = parser.parse_args()

    print('args', args)

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    results_suffix = args.results_suffix
    save_results_file = args.save_results_file
    modify_heads = args.modify_heads
    modify_all_model_heads = args.modify_all_model_heads
    modify_heads_to_values = args.modify_heads_to_values
    modify_heads_simultaneously = args.modify_heads_simultaneously
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    assert modify_heads is not None or modify_all_model_heads, \
        'Either --modify-heads or --modify-all-model-heads should be set'
    assert not modify_heads_simultaneously or not modify_all_model_heads, \
        'Cannot modify all model heads simultaneously'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AdjustableMarianMTModel.from_pretrained(model_path)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    if modify_all_model_heads:
        num_layers = model.config.encoder_layers
        num_heads = model.config.encoder_attention_heads
        heads_list = generate_full_heads_list(
            num_layers, num_heads,
            attention_types=('encoder', 'phrase_cross', 'context_cross', 'decoder', 'decoder_after')
        )

        if args.limit_layers is not None:
            heads_list = [(attention_type, layer, head)
                          for attention_type, layer, head in heads_list
                          if layer in args.limit_layers]
        if args.limit_heads is not None:
            heads_list = [(attention_type, layer, head)
                          for attention_type, layer, head in heads_list
                          if head in args.limit_heads]
    else:
        heads_list = parse_heads_list(modify_heads)

    print('heads_list', heads_list)
    print('modify_heads_to_values', modify_heads_to_values)
    print('modify_heads_simultaneously', modify_heads_simultaneously)

    all_results = []
    if modify_heads_simultaneously:
        for modify_to_value in modify_heads_to_values:
            suffix = f'mod'
            for attention_type, layer, head in heads_list:
                suffix = f'{suffix}-{attention_type}-{layer}-{head}'
            suffix = f'{suffix}-{modify_to_value}'

            if results_suffix is not None:
                suffix = f'{results_suffix}_{suffix}'

            (
                encoder_layers_heads,
                phrase_cross_attention_layer_heads,
                context_cross_attention_layer_heads,
                decoder_attention_layer_heads,
                decoder_after_attention_layer_heads
            ) = get_combined_attention_layer_heads(heads_list)

            results = score_modified_attention(
                model=model,
                tokenizer=tokenizer,
                model_name=f'{model_path} {suffix}',
                modify_to_value=modify_to_value,
                encoder_layers_heads=encoder_layers_heads,
                phrase_cross_attention_layer_heads=phrase_cross_attention_layer_heads,
                context_cross_attention_layer_heads=context_cross_attention_layer_heads,
                decoder_attention_layer_heads=decoder_attention_layer_heads,
                decoder_after_attention_layer_heads=decoder_after_attention_layer_heads,
                results_dir=args.results_dir,
                dataset_dir=args.contrapro_dir,
                dataset_context_size=args.contrapro_ctx_size,
                source_context_size=args.src_ctx_size, target_context_size=args.tgt_ctx_size,
                filter_context_size=args.filter_context_size,
                results_suffix=suffix,
                limit_size=args.limit_dataset_size, limit_plots=args.limit_plots,
                plot_separate_attentions=True, plot_separate_heads=True,
                generate_translations=args.generate_translations, max_len=args.max_len, num_beams=args.num_beams,
                save_attentions=args.save_attention_scores, batch_size=args.batch_size, )

            modified_head_name = ''
            for attention_type, layer, head in heads_list:
                modified_head_name += f'{attention_type}-{layer}-{head}'
            all_results.append({
                'modified_head': modified_head_name,
                'attention_type': 'combined',
                'layer': -1,
                'head': -1,
                'value': modify_to_value,
                'accuracy': results['accuracy'],
                'bleu': results['bleu'],
            })

    else:
        for attention_type, layer, head in heads_list:
            for modify_to_value in modify_heads_to_values:
                suffix = f'mod-{attention_type}-{layer}-{head}-{modify_to_value}'
                if results_suffix is not None:
                    suffix = f'{results_suffix}_{suffix}'

                (
                    encoder_layers_heads,
                    phrase_cross_attention_layer_heads,
                    context_cross_attention_layer_heads,
                    decoder_attention_layer_heads,
                    decoder_after_attention_layer_heads
                ) = get_attention_layer_heads(attention_type, layer, head)

                results = score_modified_attention(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=f'{model_path} {suffix}',
                    modify_to_value=modify_to_value,
                    encoder_layers_heads=encoder_layers_heads,
                    phrase_cross_attention_layer_heads=phrase_cross_attention_layer_heads,
                    context_cross_attention_layer_heads=context_cross_attention_layer_heads,
                    decoder_attention_layer_heads=decoder_attention_layer_heads,
                    decoder_after_attention_layer_heads=decoder_after_attention_layer_heads,
                    results_dir=args.results_dir,
                    dataset_dir=args.contrapro_dir,
                    dataset_context_size=args.contrapro_ctx_size,
                    source_context_size=args.src_ctx_size, target_context_size=args.tgt_ctx_size,
                    filter_context_size=args.filter_context_size,
                    results_suffix=suffix,
                    limit_size=args.limit_dataset_size, limit_plots=args.limit_plots,
                    plot_separate_attentions=True, plot_separate_heads=True,
                    generate_translations=args.generate_translations, max_len=args.max_len, num_beams=args.num_beams,
                    save_attentions=args.save_attention_scores, batch_size=args.batch_size, )

                all_results.append({
                    'modified_head': f'{attention_type}-{layer}-{head}',
                    'attention_type': attention_type,
                    'layer': layer,
                    'head': head,
                    'value': modify_to_value,
                    'accuracy': results['accuracy'],
                    'bleu': results['bleu'],
                })

    if save_results_file is not None:
        import pandas as pd

        print('Saving results to', save_results_file)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(save_results_file, index=False, sep='\t')
