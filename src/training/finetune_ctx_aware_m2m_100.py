import argparse
import os
import shutil

import datasets
import evaluate
import numpy as np
import torchinfo
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data import load_iwslt2017_dataset, concatenate_current_and_context
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer



def translate_text(tokenizer, model, text, src_context, tgt_context, sep_token, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang if src_lang in LANG_MAP.values() else LANG_MAP[src_lang]
    tokenizer.tgt_lang = tgt_lang if tgt_lang in LANG_MAP.values() else LANG_MAP[tgt_lang]

    src_input = concatenate_current_and_context(text, [src_context], sep_token)
    tgt_input = concatenate_current_and_context('', [tgt_context], sep_token)
    inputs = tokenizer(src_input, text_target=tgt_input, return_tensors='pt').to(model.device)
    tokenized_tgt_context = inputs['labels']
    tokenized_tgt_context = tokenized_tgt_context[..., :-1]  # remove </s> token from the end
    translated_tokens = model.generate(inputs['input_ids'],
                                       decoder_input_ids=tokenized_tgt_context,
                                       max_length=100)
    print(src_input)
    print(inputs['input_ids'])
    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    print(tokenizer.convert_ids_to_tokens(tokenized_tgt_context[0]))
    print(translated_tokens)
    output_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=False)[0]
    print(output_text)
    return output_text, translated_tokens


def get_sacrebleu_metric_fn(tokenizer):
    metric = evaluate.load('sacrebleu')

    def compute_metric(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {'bleu': result['score']}

    return compute_metric


def load_datasets():
    train_datasets = []
    eval_datasets = []
    test_datasets = []
    for tgt_lang in tgt_langs:
        src_lang_code = LANG_MAP[src_lang]
        tgt_lang_code = LANG_MAP[tgt_lang]
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
        print(f'Using languages: {args.src_lang} ({src_lang_code}) -> {args.tgt_lang} ({tgt_lang_code})')

        if not sample_ctx_size:
            (
                train_dataset,
                eval_dataset,
                test_dataset
            ) = load_iwslt2017_dataset(tokenizer, base_data_dir,
                                       src_lang, tgt_lang,
                                       context_sep_token=context_sep_token,
                                       src_ctx_size=src_ctx_size,
                                       tgt_ctx_size=tgt_ctx_size,
                                       max_length=max_length,
                                       model_name=model_name_short,
                                       include_forced_bos_token=True,
                                       tokenizer_language_dict=LANG_MAP,
                                       seed=1)
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            test_datasets.append(test_dataset)
        else:
            context_sizes = []
            if match_ctx_size:
                assert src_ctx_size == tgt_ctx_size
                context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
            else:
                context_sizes = [(s, t) for s in range(src_ctx_size + 1) for t in range(tgt_ctx_size + 1)]

            print(f'Loading datasets with context sizes: {context_sizes}')
            for src_cs, tgt_cs in context_sizes:
                (
                    train_dataset, eval_dataset, test_dataset, dataset
                ) = load_iwslt2017_dataset(tokenizer, base_data_dir,
                                           src_lang, tgt_lang,
                                           context_sep_token=context_sep_token,
                                           src_ctx_size=src_cs,
                                           tgt_ctx_size=tgt_cs,
                                           max_length=max_length,
                                           model_name=model_name_short,
                                           include_forced_bos_token=True,
                                           tokenizer_language_dict=LANG_MAP,
                                           seed=1)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                test_datasets.append(test_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
        eval_dataset = datasets.interleave_datasets(eval_datasets)
        test_dataset = datasets.interleave_datasets(test_datasets)
    else:
        train_dataset = train_datasets[0]
        eval_dataset = eval_datasets[0]
        test_dataset = test_datasets[0]

    return train_dataset, eval_dataset, test_dataset



parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="facebook/nllb-200-distilled-600M", type=str,
                    help='path to the model to be fine-tuned')
parser.add_argument("--tokenizer-path", default=None, type=str,
                    help='path to the tokenizer to be used (if different from the model)')
parser.add_argument("--model-name", default=None, type=str,
                    help='name of the model to be fine-tuned')
parser.add_argument("--base-data-dir", default='.', type=str,
                    help='base directory to save load data')  # '../../../data/iwslt_hf'
parser.add_argument("--src-lang", default='en', type=str, help='source language')
parser.add_argument("--tgt-lang", default='de', type=str, help='target language')
parser.add_argument("--tgt-langs", default='de', type=str, nargs='+', help='target languages')
parser.add_argument("--src-ctx-size", default=0, type=int, help='size of the source side')
parser.add_argument("--tgt-ctx-size", default=0, type=int, help='size of the target side')
parser.add_argument("--sample-ctx-size", default=False, action='store_true',
                    help='sample the size of the context')
parser.add_argument("--match-ctx-size", default=False, action='store_true',
                    help='match the sampled size of the source and target context')
parser.add_argument("--max-length", default=200, type=int, help='maximum length of the sentences')
parser.add_argument("--learning-rate", default=5e-5, type=float)
parser.add_argument("--weight-decay", default=1e-2, type=float)
parser.add_argument("--warmup-ratio", default=0.0, type=float)
parser.add_argument("--per-device-train-batch-size", default=12, type=int)
parser.add_argument("--per-device-eval-batch-size", default=12, type=int)
parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
parser.add_argument("--num-train-epochs", default=10, type=int)
parser.add_argument("--save-total-limit", default=10, type=int)
parser.add_argument("--no-sep-token", action='store_true', default=False,
                    help="if set, doesn't add the separator token")

args = parser.parse_args()

print(args)

base_data_dir = args.base_data_dir
# add_sep_token = not args.no_sep_token

model_name = args.model_path
tokenizer_name = args.tokenizer_path if args.tokenizer_path else model_name

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# if add_sep_token:
#     tokenizer.add_special_tokens({'sep_token': '</sep>'})
#     model.resize_token_embeddings(len(tokenizer))

torchinfo.summary(model)

model_name_short = args.model_name
if model_name_short is None:
    if model_name.startswith('facebook/'):
        model_name_short = model_name.split('/')[-1]
    else:
        raise ValueError('Please provide a model name')

model_name_short += f'_{args.src_lang}_'
for lang in args.tgt_langs:
    model_name_short += f'{lang}_'

run_name = f'{model_name_short}_iwslt2017'
src_lang = args.src_lang
tgt_langs = args.tgt_langs
src_ctx_size = args.src_ctx_size
tgt_ctx_size = args.tgt_ctx_size
sample_ctx_size = args.sample_ctx_size
match_ctx_size = args.match_ctx_size
max_length = args.max_length
context_sep_token = tokenizer.sep_token

if context_sep_token is None:
    print("Tokenizer does not have sep_token set!!! The empty separator will be used instead.")

LANG_MAP = {
    'en': 'eng_Latn',
    'de': 'deu_Latn',
    'fr': 'fra_Latn',
}

train_dataset, eval_dataset, test_dataset = load_datasets()

training_args = transformers.Seq2SeqTrainingArguments(
    run_name,
    logging_strategy="epoch",
    optim="adafactor",
    evaluation_strategy='no',
    eval_steps=None,
    save_strategy='epoch',
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    # gradient_checkpointing=True,
    weight_decay=args.weight_decay,
    save_total_limit=args.save_total_limit,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

translate_text(tokenizer, model,
               text=f'It is good for the kids.',
               src_context='What a beautiful house!',
               tgt_context=f'Was für ein schönes Haus!',
               sep_token=context_sep_token,
               src_lang='en',
               tgt_lang='de',
               )

compute_metric = get_sacrebleu_metric_fn(tokenizer)

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
trainer = M2MSeq2SeqTrainer(model,
                            training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metric,
                            )

print(f"Saving initial model to 'checkpoint-0'...")
model.save_pretrained('./checkpoint-0')

# pre_training_eval_results = trainer.evaluate(eval_dataset, max_length=max_length, metric_key_prefix='eval')
# print('eval', pre_training_eval_results)
# pre_training_test_results = trainer.evaluate(test_dataset, max_length=max_length, metric_key_prefix='test')
# print('test', pre_training_test_results)

print('Training started...')
trainer.train(resume_from_checkpoint=False)
print('Training complete.')

translate_text(tokenizer, model,
               text=f'It is good for the kids.',
               src_context='What a beautiful house!',
               tgt_context=f'Was für ein schönes Haus!',
               sep_token=context_sep_token,
               src_lang='en',
               tgt_lang='de',)

post_training_eval_results = trainer.evaluate(eval_dataset, max_length=max_length, metric_key_prefix='eval')
print('eval', post_training_eval_results)
post_training_test_results = trainer.evaluate(test_dataset, max_length=max_length, metric_key_prefix='test')
print('test', post_training_test_results)

torchinfo.summary(model)

translate_text(tokenizer, model,
               text=f'It is good for the kids.',
               src_context='What a beautiful house!',
               tgt_context=f'Was für ein schönes Haus!',
               sep_token=context_sep_token,
               src_lang='en',
               tgt_lang='de',)

model.save_pretrained('./model')
tokenizer.save_pretrained('./tokenizer')

# Move the initial model to the run directory, as the directory is deleted and recreated when the training starts
print(f"Saving initial model from 'checkpoint-0' to '{os.path.join(run_name, 'checkpoint-0')}'...")
shutil.move('./checkpoint-0', os.path.join(run_name, 'checkpoint-0'))

# Save args to a file
with open('args.txt', 'w') as f:
    f.write(str(args))
