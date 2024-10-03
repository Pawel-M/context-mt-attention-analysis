# Analyzing the Attention Heads for Pronoun Disambiguation in Context-aware Machine Translation Models

This is the code repository accompanying the paper "Analyzing the Attention Heads for Pronoun Disambiguation
in Context-aware Machine Translation Models".

## Requirements

The code uses ```python 3.8```. Install the required packages from the ```requirements.txt``` file.

```shell
pip install -r requirements.txt
```

## Data
We provide the data for the contrastive datasets used in this paper in the ```data``` folder. 
The data used for training is downloaded automatically from huggingface.

The data used in this paper was obtained from the following repositories:
- ContraPro contrastive dataset: https://github.com/ZurichNLP/ContraPro
- Large Contrastive Pronoun Dataset: https://github.com/rbawden/Large-contrastive-pronoun-testset-EN-FR
- ctxpro toolset/dataset: https://github.com/rewicks/ctxpro


## Context-aware Fine Tuning

To fine-tune the sentence-level MT models use the scripts below.

### OpusMT en-de

Change parameters `--src-ctx-size` and `--tgt-ctx-size` to 3 to train the model with the maximum context size of 3.
Additionally, change the `--max-length` parameter to 300. 
In the paper, we also decreased the batch size (`--per-device-train-batch-size`) to 16 but increased the `--gradient-accumulation-steps` to 16.

```shell

```shell
PYTHONPATH=path/to/repo/src/ python -m training.finetune_ctx_aware_opus_mt \
  --base-data-dir path/to/repo/data/iwslt_hf \
  --src-lang en \
  --tgt-lang de \
  --src-ctx-size 1 \
  --tgt-ctx-size 1 \
  --sample-ctx-size \
  --match-ctx-size \
  --max-length 200 \
  --learning-rate 5e-5 \
  --weight-decay 1e-2 \
  --warmup-ratio 0.0 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 10 \
  --save-total-limit 10
```

### NLLB-200 600M

```shell

```shell
PYTHONPATH=path/to/repo/src/ python -m training.finetune_ctx_aware_m2m_100 \
  --base-data-dir path/to/repo/data/iwslt_hf \
  --src-lang en \
  --tgt-lang de \
  --src-ctx-size 1 \
  --tgt-ctx-size 1 \
  --sample-ctx-size \
  --match-ctx-size \
  --max-length 200 \
  --learning-rate 5e-5 \
  --weight-decay 1e-2 \
  --warmup-ratio 0.0 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 32 \
  --num-train-epochs 10 \
  --save-total-limit 10
```

## ContraPro Scoring


### Sentence-level Opus MT on sentence-level dataset

```shell

PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_contrapro \
  --results-dir . \
  --results-suffix 'ctx_0_filtered' \
  --contrapro-dir ../../../../Datasets/ContraPro \
  --contrapro-ctx-size 0 \
  --filter-context-size \
  --src-lang en \
  --tgt-lang de \
  --src-ctx-size 0 \
  --tgt-ctx-size 0 \
  --model-path "Helsinki-NLP/opus-mt-en-de" \
  --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
  --limit-plots 20 \
  --generate-translations
```

### Sentence-level NLLB-600M

```shell
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_contrapro   \
--results-dir .    \
--contrapro-dir ~/Datasets/ContraPro   \
--contrapro-ctx-size 0     \
--filter-context-size   \
--src-lang en     \
--tgt-lang de     \
--src-ctx-size 0     \
--tgt-ctx-size 0     \
--model-path "facebook/nllb-200-distilled-600M"     \
--tokenizer-path "facebook/nllb-200-distilled-600M"     \
--limit-plots 0     \
--max-length 200   \
--generate-translations   \
--batch-size 500 > nohup_1.txt &
```


### Context-aware Opus MT

```shell
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../src/ nohup python -m evaluating.opus_mt_contrapro \
  --results-dir . \
  --contrapro-dir ../../../../Datasets/ContraPro \
  --contrapro-ctx-size 1 \
  --src-lang en \
  --tgt-lang de \
  --src-ctx-size 1 \
  --tgt-ctx-size 1 \
  --model-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-5e-6/model/ \
  --tokenizer-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-5e-6/tokenizer/ \
  --limit-plots 20 \
  --generate-translations \
  --batch-size 2000  > nohup_3.txt &
```

### Context-aware NLLB-600M

```shell
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_contrapro   \
--results-dir .    \
--contrapro-dir ~/Datasets/ContraPro   \
--contrapro-ctx-size 1     \
--src-lang en     \
--tgt-lang de     \
--src-ctx-size 1     \
--tgt-ctx-size 1     \
--model-path ../checkpoint_best     \
--tokenizer-path ../tokenizer     \
--limit-plots 20     \
--max-length 200   \
--generate-translations   \
--batch-size 500 > nohup_4.txt &
```

## LCPT scoring


### Sentence-level NLLB-600M

```shell
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_contrapro   \
--results-dir .    \
--contrapro-dir ~/Datasets/Large-contrastive-pronoun-testset-EN-FR/   \
--contrapro-ctx-size 0     \
--filter-context-size   \
--src-lang en     \
--tgt-lang fr     \
--src-ctx-size 0     \
--tgt-ctx-size 0     \
--model-path "facebook/nllb-200-distilled-600M"     \
--tokenizer-path "facebook/nllb-200-distilled-600M"     \
--limit-plots 0     \
--max-length 200   \
--generate-translations   \
--batch-size 500 > nohup_1.txt &
```

#### Ctx 1 NLLB-600M LCPT

```shell
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_contrapro   \
--results-dir .    \
--contrapro-dir ~/attention-analysis/data/LCPT   \
--contrapro-ctx-size 1     \
--src-lang en     \
--tgt-lang fr     \
--src-ctx-size 1     \
--tgt-ctx-size 1     \
--model-path ../checkpoint_best     \
--tokenizer-path ../tokenizer     \
--limit-plots 0     \
--max-length 200   \
--generate-translations   \
--batch-size 500 > nohup_4.txt &
```


## Translating

```shell
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=../../../src/ nohup python -m evaluating.opus_mt_translate \
--model-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/checkpoint_best \
--tokenizer-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/tokenizer \
--model-name opus_mt \
--results-dir . \
--results-suffix checkpoint_best \
--dataset ContraPro \
--base-data-dir ../../../data/ContraPro_trans_hf \
--raw-data-dir ../../../../Datasets/ContraPro \
--src-lang en \
--tgt-lang de \
--src-ctx-size 1 \
--tgt-ctx-size 1 \
--max-length 200 \
--batch-size 32 > nohup_6.txt &
```

```shell
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_translate \
--model-path "facebook/nllb-200-distilled-600M" \
--tokenizer-path "facebook/nllb-200-distilled-600M" \
--model-name nllb \
--results-dir . \
--dataset IWSLT \
--base-data-dir ../../../data/ContraPro_trans_hf \
--raw-data-dir ../../../../Datasets/ContraPro \
--src-lang en \
--tgt-lang de \
--src-ctx-size 1 \
--tgt-ctx-size 1 \
--max-length 200 \
--batch-size 1 > nohup_6.txt &
```

```shell
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_translate \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --model-name "nllb" \
    --results-dir . \
    --results-suffix "en-de.valid" \
    --dataset iwslt2017 \
    --base-data-dir ~/attention-analysis/data/iwslt2017_hf \
    --dataset-splits "valid" \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --max-length 200 \
    --batch-size 1 > nohup_2.txt &
```

## Disabling Heads

The script below disables all heads of the model and saves the results to the `disabled_heads_results.tsv` file.

### Sentence-level Opus MT on sentence-level dataset

```shell
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=../../../../src/ nohup python -m evaluating.opus_mt_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir ../../../../../Datasets/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 0 \
    --disable-all-model-heads > nohup_7.txt &
```

### Sentence-level NLLB-600M on sentence-level (filtered ctx-0) datset

```shell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir ~/Datasets/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --limit-plots 0 \
    --batch-size 500 \
    --disable-all-model-heads > nohup_0.txt &
```

### Fine-tuned Opus MT Ctx 1

```shell
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../src/ nohup python -m evaluating.opus_mt_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir ../../../../../Datasets/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path ../../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/checkpoint_best/ \
    --tokenizer-path ../../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/tokenizer/ \
    --limit-plots 0 \
    --disable-all-model-heads > nohup_1.txt &
```

#### Fine-tuned Opus MT Ctx 3

```shell
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.opus_mt_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir ~/Datasets/ContraPro \
    --contrapro-ctx-size 3 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 3 \
    --tgt-ctx-size 3 \
    --model-path ../checkpoint_best \
    --tokenizer-path ../tokenizer/ \
    --limit-plots 0 \
    --batch-size 2000 \
    --disable-all-model-heads > nohup_5.txt &
```


#### Ctx 1 NLLB-600M ContraPro

```shell
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_disable_heads_contrapro   \
    --results-dir .    \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir ~/Datasets/ContraPro   \
    --contrapro-ctx-size 1     \
    --src-lang en     \
    --tgt-lang de     \
    --src-ctx-size 1     \
    --tgt-ctx-size 1     \
    --model-path ../../checkpoint_best     \
    --tokenizer-path ../../tokenizer     \
    --limit-plots 0     \
    --max-length 200   \
    --batch-size 500 \
    --disable-all-model-heads > nohup_5.txt &
```


## Modifying attentions


### Modifying specific heads

Parameter `--modify-heads-simultaneously` is used to modify all heads simultaneously.

```shell
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=../../../../src/ nohup python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --contrapro-dir ../../../../../Datasets/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 1 \
    --limit-dataset-size 100 \
    --modify-heads "encoder,5,0" "phrase_cross,5,0" "context_cross,4,0" "decoder_after,5,3" "decoder_after,5,5" \
    --modify-heads-to-values 0.01 0.99 \
    --modify-heads-simultaneously > nohup_6.txt &
```

### Modify all heads of the model


#### Sentence-level Opus MT on sentence-level dataset

```shell
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=../../../../src/ nohup python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir ../../../../../Datasets/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 0 \
    --modify-all-model-heads \
    --modify-heads-to-values 0.01 0.25 0.5 0.75 0.99 > nohup_5.txt &
```

#### Fine-tuned Opus MT Ctx 1

```shell

#    --limit-layers 3 4 5 \
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=../../../../src/ nohup python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir ../../../../../Datasets/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path ../../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/checkpoint_best/ \
    --tokenizer-path ../../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/tokenizer/ \
    --limit-plots 0 \
    --modify-all-model-heads \
    --limit-layers 4 5 \
    --batch-size 2000 \
    --modify-heads-to-values 0.01 0.25 0.5 0.75 0.99 > nohup_7.txt &
```


#### Fine-tuned Opus MT Ctx 3

```shell

#    --limit-layers 3 4 5 \
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir ~/Datasets/ContraPro \
    --contrapro-ctx-size 3 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 3 \
    --tgt-ctx-size 3 \
    --model-path ../checkpoint_best/ \
    --tokenizer-path ../tokenizer/ \
    --limit-plots 0 \
    --modify-all-model-heads \
    --batch-size 2000 \
    --modify-heads-to-values 0.01 > nohup_4.txt &
```

#### Sentence-level NLLB-200

```shell
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=~/attention-analysis/src/ nohup python -m evaluating.m2m_100_modify_attention_contrapro     \
  --results-dir .     \
  --save-results-file modified_heads_results.tsv     \
  --contrapro-dir ~/attention-analysis/data/LCPT     \
  --contrapro-ctx-size 0     \
  --filter-context-size     \
  --src-lang en     \
  --tgt-lang fr     \
  --src-ctx-size 0     \
  --tgt-ctx-size 0     \
  --model-path "facebook/nllb-200-distilled-600M"     \
  --tokenizer-path "facebook/nllb-200-distilled-600M"     \
  --limit-plots 0     \
  --modify-all-model-heads     \
  --batch-size 1000     \
  --limit-layers 0 1 2 3 4 5      \
  --modify-heads-to-values 0.01 > nohup_7.txt &

```

## Fine-tune Heads

### Sentence-level Opus MT

```shell
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../src/ nohup python -m head_tuning.opus_mt_head_tuning \
--student-model-path Helsinki-NLP/opus-mt-en-de \
--src-lang en \
--tgt-lang de \
--src-ctx-size 0 \
--tgt-ctx-size 0 \
--dataset contrapro \
--base-dataset-dir ../../../../Datasets/ContraPro \
--processed-dataset-dir ../../../data/ContraPro_hf/opus_mt_ctx_0 \
--test-size 0.5 \
--split-seed 1 \
--tuned-heads "decoder,5,3" \
--optimizer adafactor \
--tuning-loss mse \
--lambda-prediction 0.0 \
--lambda-head-tune 1.0 \
--lambda-head-stabilize 1.0 \
--freeze-non-qk \
--max-length  200 \
--max-token-idx-length 20 \
--learning-rate 5e-5 \
--weight-decay 1e-2 \
--warmup-ratio 0.0 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--gradient-accumulation-steps 8 \
--use-fp16 \
--num-train-epochs 10 \
--save-total-limit 10 > nohup_3.txt &
```

### Context-aware Opus MT

```shell
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../../../src/ nohup python -m head_tuning.opus_mt_head_tuning \
--student-model-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/checkpoint_best \
--tokenizer-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/tokenizer \
--src-lang en \
--tgt-lang de \
--src-ctx-size 1 \
--tgt-ctx-size 1 \
--dataset contrapro \
--base-dataset-dir ../../../../Datasets/ContraPro \
--processed-dataset-dir ../../../data/ContraPro_hf/opus_mt_ctx_1 \
--test-size 0.5 \
--split-seed 1 \
--tuned-heads "decoder,5,3" \
--optimizer adafactor \
--tuning-loss mse \
--lambda-prediction 0.0 \
--lambda-head-tune 1.0 \
--lambda-head-stabilize 1.0 \
--freeze-non-qk \
--max-length  200 \
--max-token-idx-length 20 \
--learning-rate 5e-5 \
--weight-decay 1e-2 \
--warmup-ratio 0.0 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--gradient-accumulation-steps 8 \
--use-fp16 \
--num-train-epochs 10 \
--save-total-limit 10 > nohup_4.txt &
```


## Fine-tune Heads

### Sentence-level Opus MT

```shell
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../src/ nohup python -m head_tuning.opus_mt_head_tuning \
--student-model-path Helsinki-NLP/opus-mt-en-de \
--src-lang en \
--tgt-lang de \
--src-ctx-size 0 \
--tgt-ctx-size 0 \
--dataset contrapro \
--base-dataset-dir ../../../../Datasets/ContraPro \
--processed-dataset-dir ../../../data/ContraPro_hf/opus_mt_ctx_0 \
--test-size 0.5 \
--split-seed 1 \
--tuned-heads "decoder,5,3" \
--optimizer adafactor \
--tuning-loss mse \
--lambda-prediction 0.0 \
--lambda-head-tune 1.0 \
--lambda-head-stabilize 1.0 \
--freeze-non-qk \
--max-length  200 \
--max-token-idx-length 20 \
--learning-rate 5e-5 \
--weight-decay 1e-2 \
--warmup-ratio 0.0 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--gradient-accumulation-steps 8 \
--use-fp16 \
--num-train-epochs 10 \
--save-total-limit 10 > nohup_3.txt &
```

### Context-aware Opus MT

```shell
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../../../src/ nohup python -m head_tuning.opus_mt_head_tuning \
--student-model-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/checkpoint_best \
--tokenizer-path ../../iwslt17_finetuned/opus_mt_en_de_1-1-sample-match_e-10_bs-32_lr-1e-5/tokenizer \
--src-lang en \
--tgt-lang de \
--src-ctx-size 1 \
--tgt-ctx-size 1 \
--dataset contrapro \
--base-dataset-dir ../../../../Datasets/ContraPro \
--processed-dataset-dir ../../../data/ContraPro_hf/opus_mt_ctx_1 \
--test-size 0.5 \
--split-seed 1 \
--tuned-heads "decoder,5,3" \
--optimizer adafactor \
--tuning-loss mse \
--lambda-prediction 0.0 \
--lambda-head-tune 1.0 \
--lambda-head-stabilize 1.0 \
--freeze-non-qk \
--max-length  200 \
--max-token-idx-length 20 \
--learning-rate 5e-5 \
--weight-decay 1e-2 \
--warmup-ratio 0.0 \
--per-device-train-batch-size 12 \
--per-device-eval-batch-size 12 \
--gradient-accumulation-steps 8 \
--use-fp16 \
--num-train-epochs 10 \
--save-total-limit 10 > nohup_4.txt &
```


## AI Assistant Usage

During writing the code in this repository we used Github Copilot (https://github.com/features/copilot).