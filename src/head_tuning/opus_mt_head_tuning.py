import argparse
import functools
import os
import re
import shutil

import evaluate
import numpy as np
import torchinfo
from transformers import (
    AutoTokenizer
)

from data import load_contrapro_dataset_raw
from data.contrapro_training_dataset import ContraPro
from evaluating.opus_mt_translate import load_dataset_and_translate
from evaluating.utils import parse_heads_list
from head_tuning.head_tune_data import (
    HeadTuneDataCollatorForSeq2Seq,
    preprocess_function
)
from head_tuning.head_tune_trainer import (
    HeadTuneSeq2SeqTrainer,
    HeadTuneSeq2SeqTrainingArguments,
)
from modeling.opus_mt_adjustable import AdjustableMarianMTModel
# from modeling.opus_mt_tokenization import tokenize_and_analyze

from evaluating.opus_mt_contrapro import score as contrapro_score
from evaluating.common_tokenization import tokenize_and_analyze


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def freeze_non_qk(model, heads_list):
    # model.encoder.layers.0.self_attn.k_proj.weight
    # model.encoder.layers.0.self_attn.k_proj.bias
    # model.encoder.layers.0.self_attn.q_proj.weight
    # model.encoder.layers.0.self_attn.q_proj.bias

    # model.decoder.layers.2.self_attn.k_proj.weight
    # model.decoder.layers.2.self_attn.k_proj.bias
    # model.decoder.layers.2.self_attn.v_proj.weight
    # model.decoder.layers.2.self_attn.v_proj.bias
    # model.decoder.layers.2.self_attn.q_proj.weight
    # model.decoder.layers.2.self_attn.q_proj.bias
    # model.decoder.layers.2.self_attn.out_proj.weight
    # model.decoder.layers.2.self_attn.out_proj.bias

    # model.decoder.layers.2.encoder_attn.k_proj.weight
    # model.decoder.layers.2.encoder_attn.k_proj.bias
    # model.decoder.layers.2.encoder_attn.v_proj.weight
    # model.decoder.layers.2.encoder_attn.v_proj.bias
    # model.decoder.layers.2.encoder_attn.q_proj.weight
    # model.decoder.layers.2.encoder_attn.q_proj.bias
    # model.decoder.layers.2.encoder_attn.out_proj.weight
    # model.decoder.layers.2.encoder_attn.out_proj.bias

    # tuned_attention in ("encoder", "phrase_cross", "context_cross", "decoder", "decoder_after"

    for attention, layer, head in heads_list:
        for n, p in model.named_parameters():
            if attention == 'encoder':
                match_k = rf'model.encoder.layers.{layer}.self_attn.k_proj.*'
                match_q = rf'model.encoder.layers.{layer}.self_attn.q_proj.*'
            elif attention in ('phrase_cross', 'context_cross'):
                match_k = rf'model.decoder.layers.{layer}.encoder_attn.k_proj.*'
                match_q = rf'model.decoder.layers.{layer}.encoder_attn.q_proj.*'
            else:  # 'decoder', 'decoder_after'
                match_k = rf'model.decoder.layers.{layer}.self_attn.k_proj.*'
                match_q = rf'model.decoder.layers.{layer}.self_attn.q_proj.*'

            if re.match(match_q, n) or re.match(match_k, n):
                p.requires_grad = True
                print(f'unfreezing: {n}')
            else:
                p.requires_grad = False


def load_models(student_model_name, teacher_model_name, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_input_names=["input_ids", "token_type_ids", "attention_mask",
                           'src_context_idx', 'src_phrase_idx', 'tgt_context_idx', 'tgt_phrase_idx']
    )
    student_model = AdjustableMarianMTModel.from_pretrained(student_model_name)
    teacher_model = AdjustableMarianMTModel.from_pretrained(teacher_model_name)
    teacher_model.eval()
    return student_model, teacher_model, tokenizer


def load_dataset(dataset,
                 dataset_dir,
                 processed_dataset_dir,
                 src_lang, tgt_lang,
                 max_length, max_token_idx_length,
                 source_context_size, target_context_size,
                 test_size, split_seed, return_test=True):
    if dataset == 'contrapro':
        return load_contrapro(dataset_dir, processed_dataset_dir, src_lang, tgt_lang, max_length, max_token_idx_length,
                              source_context_size, target_context_size, test_size, split_seed, return_test)
    elif dataset == 'ctxpro':
        return load_ctxpro(dataset_dir, processed_dataset_dir, src_lang, tgt_lang, max_length, max_token_idx_length,
                           source_context_size, target_context_size, test_size, split_seed, return_test)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def load_ctxpro(dataset_dir,
                processed_dataset_dir,
                src_lang, tgt_lang,
                max_length, max_token_idx_length,
                source_context_size, target_context_size,
                test_size, split_seed, return_test=True):
    print(f'Loading ctxpro dataset from {dataset_dir}...')
    max_context_size = max(source_context_size, target_context_size)
    full_dataset_dir = os.path.join(dataset_dir, f'ctx_{max_context_size}')
    ds_builder = ContraPro(full_dataset_dir,
                           base_path=dataset_dir,
                           ctx_size=max_context_size,
                           files_base_name='ctxpro',
                           tgt_phrase_key='expected')
    ds_builder.download_and_prepare(processed_dataset_dir)
    ds = ds_builder.as_dataset('train')

    print(f'Loaded ctxpro dataset with {len(ds)} examples')

    preprocess_fn = functools.partial(
        preprocess_function,
        tokenize_and_analyze_fn=tokenize_and_analyze,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        max_token_idx_length=max_token_idx_length,
        source_context_size=source_context_size,
        target_context_size=target_context_size,
        consider_upper_phrases=True,
    )

    if test_size > 0:
        ds = ds.train_test_split(test_size=test_size, seed=split_seed)
        train_ds = ds['train']
        test_ds = ds['test']
        train_ids = ds['train']['id']
        test_ids = ds['test']['id']
    else:
        train_ds = ds
        test_ds = None
        train_ids = ds['id']
        test_ids = []

    print('Processing train dataset...')
    tokenized_train = train_ds.map(preprocess_fn, batched=True)

    if not return_test or test_ds is None:
        return tokenized_train, None, train_ids, test_ids

    print('Processing test dataset...')
    tokenized_test = test_ds.map(preprocess_fn, batched=True)

    return tokenized_train, tokenized_test, train_ids, test_ids


def load_contrapro(dataset_dir,
                   processed_dataset_dir,
                   src_lang, tgt_lang,
                   max_length, max_token_idx_length,
                   source_context_size, target_context_size,
                   test_size, split_seed, return_test=True):
    ds = load_contrapro_dataset_raw(dataset_dir, processed_dataset_dir,
                                    source_context_size, target_context_size,
                                    test_size, split_seed)

    preprocess_fn = functools.partial(
        preprocess_function,
        tokenize_and_analyze_fn=tokenize_and_analyze,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        max_token_idx_length=max_token_idx_length,
        source_context_size=source_context_size,
        target_context_size=target_context_size,
    )

    train_ids = ds['train']['id']
    test_ids = ds['test']['id']
    train_ds = ds['train']
    test_ds = ds['test']
    print('Processing train dataset...')
    tokenized_train = train_ds.map(preprocess_fn, batched=True)

    if not return_test:
        return tokenized_train, None, train_ids, test_ids

    print('Processing test dataset...')
    tokenized_test = test_ds.map(preprocess_fn, batched=True)

    return tokenized_train, tokenized_test, train_ids, test_ids


def get_run_name(model_name, heads_list, args):
    run_name = f'{model_name}_head_tune_{args.dataset}-{(1 - args.test_size)}' \
               f'_lr-{args.learning_rate}' \
               f'_e-{args.num_train_epochs}'

    run_name += f'_{heads_list[0][0]}'

    for attention_type, layer, head in heads_list:
        run_name += f'_{layer}-{head}'

    run_name += f'_{args.tuning_loss}' \
                f'_lambdas-{args.lambda_prediction}-{args.lambda_head_tune}-{args.lambda_head_stabilize}'

    if args.per_device_train_batch_size != 12 or args.gradient_accumulation_steps != 8:
        run_name += f'_bs-{args.per_device_train_batch_size}-{args.gradient_accumulation_steps}'

    if args.freeze_non_qk:
        run_name += f'_frozen'

    return run_name


def save_ids(ids, file_path):
    with open(file_path, 'w') as f:
        for id in ids:
            f.write(f'{id}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model-path", required=True, type=str,
                        help='model name (from huggingface) or model path for the student model')
    parser.add_argument("--teacher-model-path", default=None, type=str,
                        help='model name (from huggingface) or model path for the teacher model '
                             '(defaults to --student-model-path)')
    parser.add_argument("--tokenizer-path", default=None, type=str,
                        help='model name (from huggingface) or model path (defaults to --student-model-path)')
    parser.add_argument("--src-lang", default='en', type=str, help='source language')
    parser.add_argument("--tgt-lang", default='de', type=str, help='target language')
    parser.add_argument("--src-ctx-size", default=0, type=int, help='size of the source side')
    parser.add_argument("--tgt-ctx-size", default=0, type=int, help='size of the target side')
    # parser.add_argument("--sample-ctx-size", default=False, action='store_true',
    #                     help='sample the size of the context')
    # parser.add_argument("--match-ctx-size", default=False, action='store_true',
    #                     help='match the sampled size of the source and target context')
    parser.add_argument("--dataset", default='contrapro', type=str, help='dataset to use: "contrapro", "ctxpro')
    parser.add_argument("--base-dataset-dir", default='.', type=str,
                        help='base directory to load/save the dataset')  # '../../../data/ContraPro'
    parser.add_argument("--processed-dataset-dir", default='.', type=str,
                        help='directory to save the processed dataset')  # '../../../data/ContraPro_hf_ctx0'
    parser.add_argument("--test-size", default=0.5, type=float, help='size of the test split of the dataset')
    parser.add_argument("--split-seed", default=1, type=int, help='seed for the dataset split')
    parser.add_argument("--tuned-heads", default=None, type=str, nargs='+',
                        help='list of heads to modify in the form (attention_relation, layer, head), '
                             'where attention_relation is one of: '
                             '"encoder", "phrase_cross", "context_cross", "decoder", "decoder_after"')
    parser.add_argument("--optimizer", default='adafactor', type=str, help='optimizer to use')
    parser.add_argument("--tuning-loss", default='mse', type=str,
                        help='loss function to use for tuning: "mse", "kl", "mse_post"')
    parser.add_argument("--lambda-prediction", default=0.0, type=float, help='weight of the prediction loss')
    parser.add_argument("--lambda-head-tune", default=1.0, type=float, help='weight of the head tuning loss')
    parser.add_argument("--lambda-head-stabilize", default=1.0, type=float,
                        help='weight of the head stabilization loss')
    parser.add_argument("--freeze-non-qk", action='store_true', default=False,
                        help='if set, freezes the model weights apart from the tuned heads '
                             '(Q and K projection weights)')
    parser.add_argument("--max-length", default=300, type=int, help='maximum length of the sentences')
    parser.add_argument("--max-token-idx-length", default=20, type=int, help='maximum length of the token indices')
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-2, type=float)
    parser.add_argument("--warmup-ratio", default=0.0, type=float)
    parser.add_argument("--per-device-train-batch-size", default=12, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=12, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--use-fp16", action='store_true', default=False, help='if set, uses fp16 for training')
    parser.add_argument("--num-train-epochs", default=10, type=int)
    parser.add_argument("--save-strategy", default='epoch', type=str,
                        help='strategy for saving the model ("epoch", "steps")')
    parser.add_argument("--save-steps", default=100, type=int, help='number of steps between saving the model')
    parser.add_argument("--save-total-limit", default=1, type=int)
    parser.add_argument("--eval-strategy", default='epoch', type=str,
                        help='evaluation strategy: "epoch", "steps", "no"')
    parser.add_argument("--eval-steps", default=None, type=int, help='number of steps between evaluations')
    parser.add_argument("--evaluate-after-training", default=False, action='store_true',
                        help='if set, evaluates the model after training')
    parser.add_argument("--contrapro-after-training", default=False, action='store_true',
                        help='if set, evaluates the model on ContraPro after training')
    parser.add_argument("--contrapro-dir", default='.', type=str,
                        help='directory with ContraPro dataset, expects ot have ctx1, ctx2, etc. folders inside')
    parser.add_argument("--contrapro-batch-size", default=200, type=int, help='batch size for contrapro scoring')
    parser.add_argument("--num-beams", default=5, type=int, help='number of beams in generating translations')
    parser.add_argument("--translate-after-training", default=False, action='store_true',
                        help='if set, translates the dataset after training')
    parser.add_argument("--translate-dataset-name", default='iwslt2017', type=str, help='the dataset to translate')
    parser.add_argument("--translate-dataset-splits", default=None, type=str, nargs='+',
                        help='the dataset splits to translate')
    parser.add_argument("--translate-base-data-dir", default=None, type=str,
                        help='base directory to save loaded data')
    parser.add_argument("--translate-raw-data-dir", default='.', type=str,
                        help='base directory to load raw data (eg. ContraPro or ctxpro dataset dir)')
    parser.add_argument("--wandb-project", default=None, type=str, help='wandb project name')

    args = parser.parse_args()
    print(args)

    if args.wandb_project is not None:
        # if 'WANDB_API_KEY' not in os.environ:
        #     raise ValueError('WANDB_API_KEY is not set')
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    student_model_name = args.student_model_path
    teacher_model_name = args.teacher_model_path if args.teacher_model_path is not None else student_model_name
    tokenizer_name = args.tokenizer_path if args.tokenizer_path is not None else student_model_name

    student_model, teacher_model, tokenizer = load_models(student_model_name, teacher_model_name, tokenizer_name)

    evaluate_after_training = args.evaluate_after_training
    eval_strategy = args.eval_strategy
    test_size = args.test_size
    if test_size <= 0:
        eval_strategy = 'no'
        evaluate_after_training = False
    test_set_needed = args.evaluate_after_training or args.eval_strategy != 'no'

    (
        tokenized_train,
        tokenized_test,
        train_ids,
        test_ids
    ) = load_dataset(
        args.dataset,
        args.base_dataset_dir,
        args.processed_dataset_dir,
        args.src_lang, args.tgt_lang,
        args.max_length, args.max_token_idx_length,
        args.src_ctx_size, args.tgt_ctx_size,
        test_size, args.split_seed,
        return_test=test_set_needed)

    print('train dataset:', len(tokenized_train))

    print('Printing examples without target phrases:')
    for i in range(len(tokenized_train)):
        if all([t == -1 for t in tokenized_train[i]['tgt_phrase_idx']]):
            print(i)
            print(tokenized_train[i])

    tuned_heads = args.tuned_heads
    heads_list = parse_heads_list(tuned_heads)
    assert all([at == heads_list[0][0] for at, _, _ in heads_list]), 'Different attention types are not supported yet'
    tuned_attention = heads_list[0][0]

    if args.freeze_non_qk:
        freeze_non_qk(student_model, heads_list)

    run_name = get_run_name('opus_mt', heads_list, args)

    data_collator = HeadTuneDataCollatorForSeq2Seq(tokenizer=tokenizer, model=student_model_name)

    metric = evaluate.load("sacrebleu")

    additional_args = ['src_context_idx', 'src_phrase_idx', 'tgt_context_idx', 'tgt_phrase_idx']
    training_args = HeadTuneSeq2SeqTrainingArguments(
        run_name,
        report_to="wandb" if args.wandb_project is not None else None,
        logging_strategy="epoch",
        optim=args.optimizer,
        evaluation_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_checkpointing=True,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=args.use_fp16,
        push_to_hub=False,
        tuned_heads=heads_list,
        # tuned_attention=tuned_attention,
        # tuned_token_type=tuned_token_type,
        # tuned_layer_heads=tuned_layer_heads,
        # tuned_head_layer=tuned_head_layer,
        # tuned_head_index=tuned_head_index,
        lambda_prediction=args.lambda_prediction,
        lambda_head_tune=args.lambda_head_tune,
        lambda_head_stabilize=args.lambda_head_stabilize,
        label_names=additional_args,
        tune_loss_type=args.tuning_loss,
        ignore_index=tokenizer.pad_token_id,
        # generation_config=HeadTuneGenerationConfig.from_pretrained(student_model_name, ignore_model_kwargs=additional_args),
    )

    torchinfo.summary(student_model)

    trainer = HeadTuneSeq2SeqTrainer(
        model=student_model,
        teacher=teacher_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test if args.eval_strategy != 'no' else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    checkpoint_0_name = f'{run_name}_checkpoint-0'
    print(f"Saving initial model to '{checkpoint_0_name}'...")
    student_model.save_pretrained(checkpoint_0_name)

    print(f'Starting training for run: {run_name}')
    trainer.train(
        resume_from_checkpoint=False,
        # ignore_keys_for_eval=[tokenizer(tokenizer.tgt_lang)],
    )

    print(f'Saving final model to {os.path.join(run_name, "final_model")}...')
    student_model.save_pretrained(os.path.join(run_name, 'final_model'))

    print(f'Saving tokenizer to {os.path.join(run_name, "tokenizer")}...')
    tokenizer.save_pretrained(os.path.join(run_name, 'tokenizer'))

    print(f"Saving initial model from '{checkpoint_0_name}' to '{os.path.join(run_name, 'checkpoint-0')}'...")
    shutil.move(checkpoint_0_name, os.path.join(run_name, 'checkpoint-0'))

    save_ids(train_ids, os.path.join(run_name, 'train_ids.txt'))
    save_ids(test_ids, os.path.join(run_name, 'test_ids.txt'))

    with open(os.path.join(run_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    if evaluate_after_training:
        # train_results = trainer.evaluate(tokenized_train)
        # print('train post results:', train_results)

        test_results = trainer.evaluate(tokenized_test)
        print('test post results:', test_results)

    if args.contrapro_after_training:
        student_model.eval()
        contrapro_ctx_size = max(args.src_ctx_size, args.tgt_ctx_size)
        results = contrapro_score(
            model=student_model,
            tokenizer=tokenizer,
            model_name=run_name,
            results_dir=run_name,
            dataset_dir=args.contrapro_dir,
            dataset_context_size=contrapro_ctx_size,
            source_context_size=args.src_ctx_size, target_context_size=args.tgt_ctx_size,
            filter_context_size=contrapro_ctx_size == 0,
            results_suffix='final_model',
            limit_size=None, limit_plots=0,
            plot_separate_attentions=False, plot_separate_heads=False,
            generate_translations=True,
            max_len=args.max_length, num_beams=args.num_beams,
            save_attentions=False,
            batch_size=args.contrapro_batch_size,
        )

    # parser.add_argument("--translate-after-training", default=False, action='store_true',
    #                     help='if set, translates the dataset after training')
    # parser.add_argument("--translate-dataset-name", default='iwslt2017', type=str, help='the dataset to translate')
    # parser.add_argument("--translate-dataset-splits", default=None, type=str, nargs='+',
    #                     help='the dataset splits to translate')
    # parser.add_argument("--translate-base-data-dir", default=None, type=str,
    #                     help='base directory to save loaded data')
    # parser.add_argument("--translate-raw-data-dir", default='.', type=str,
    #                     help='base directory to load raw data (eg. ContraPro or ctxpro dataset dir)')

    if args.translate_after_training:
        student_model.eval()
        bleus = load_dataset_and_translate(
            student_model, tokenizer, 'opus_mt',
            args.translate_dataset_name,
            args.translate_base_data_dir, args.translate_raw_data_dir,
            args.src_lang, args.tgt_lang,
            args.src_ctx_size, args.tgt_ctx_size,
            # args.sample_ctx_size, args.match_ctx_size,
            args.num_beams, args.max_length,
            run_name,
            dataset_splits=args.translate_dataset_splits,
            results_suffix='final_model',
            batch_size=16,
        )
