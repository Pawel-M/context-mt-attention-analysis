import argparse
import functools

import torch
from transformers import AutoTokenizer, MarianTokenizer

from evaluating.contrapro_score import score_and_plot_contrapro
from evaluating.common_functions import score_contrastive
from evaluating.m2m_100_functions import generate_translation
from modeling.m2m100_adjustable import AdjustableM2M100ForConditionalGeneration


def score(model, tokenizer, results_dir, dataset_dir, model_name,
          dataset_context_size=None, source_context_size=None, target_context_size=None,
          filter_context_size=False,
          is_contrapro_dataset=True,
          limit_ids=None, limit_size=None, limit_plots=200,
          results_suffix=None,
          plot_separate_attentions=False, plot_separate_heads=False,
          generate_translations=True, max_len=300, num_beams=5,
          save_attentions=True, batch_size=None):
    source_context_size = 0 if source_context_size is None else source_context_size
    target_context_size = 0 if target_context_size is None else target_context_size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scorer = functools.partial(score_contrastive, model=model,
                               source_context_size=source_context_size, target_context_size=target_context_size,
                               tokenizer=tokenizer, device=device, consider_upper_phrases=not is_contrapro_dataset)

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
        use_json_lines=is_contrapro_dataset,
        limit_ids=limit_ids,
        limit_size=limit_size,
        limit_plots=limit_plots,
        results_suffix=results_suffix,
        plot_separate_attentions=plot_separate_attentions,
        plot_separate_heads=plot_separate_heads,
        generate_translations=generate_translations,
        save_results_to_file=True,
        save_attentions=save_attentions,
        batch_size=batch_size,
    )
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default='.', type=str, help='directory to save the results')
    parser.add_argument("--results-suffix", default=None, type=str, help='suffix added to the results file')
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
    parser.add_argument("--max-length", default=300, type=int, help='maximum length of the generated translations')
    parser.add_argument("--num-beams", default=5, type=int, help='number of beams in generating translations')
    parser.add_argument("--not-save-attention-scores", action='store_true', default=False,
                        help='if set, does not save attention scores to a file')
    parser.add_argument("--batch-size", default=None, type=int, help='batch size for scoring')

    args = parser.parse_args()

    print('args', args)

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AdjustableM2M100ForConditionalGeneration.from_pretrained(model_path)

    LANG_MAP = {
        'en': 'eng_Latn',
        'de': 'deu_Latn',
        'fr': 'fra_Latn',
    }
    src_lang_code = LANG_MAP[args.src_lang]
    tgt_lang_code = LANG_MAP[args.tgt_lang]
    tokenizer.src_lang = src_lang_code
    tokenizer.tgt_lang = tgt_lang_code
    print(f'Using languages: {args.src_lang} ({src_lang_code}) -> {args.tgt_lang} ({tgt_lang_code})')

    print('tokenizer', tokenizer)
    print('tokenizer special tokens:', tokenizer.special_tokens_map)
    print('model', model)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    use_contrapro_dataset = 'contrapro' in args.contrapro_dir.lower()
    print('use_contrapro_dataset', use_contrapro_dataset)

    results = score(
        model=model,
        tokenizer=tokenizer,
        model_name=model_path,
        results_dir=args.results_dir,
        dataset_dir=args.contrapro_dir,
        dataset_context_size=args.contrapro_ctx_size,
        source_context_size=args.src_ctx_size, target_context_size=args.tgt_ctx_size,
        filter_context_size=args.filter_context_size,
        is_contrapro_dataset=use_contrapro_dataset,
        results_suffix=args.results_suffix,
        limit_size=args.limit_dataset_size, limit_plots=args.limit_plots,
        plot_separate_attentions=True, plot_separate_heads=True,
        generate_translations=args.generate_translations,
        max_len=args.max_length, num_beams=args.num_beams,
        save_attentions=not args.not_save_attention_scores,
        batch_size=args.batch_size,
    )
