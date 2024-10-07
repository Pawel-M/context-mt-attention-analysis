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

## ContraPro and LCPT Scoring

The following shows how to score the models on the ContraPro/LCPT datasets. 

### Sentence-level OpusMT en-de

The parameter ```--filter-context-size``` if set, is used to filter the dataset 
to contain only the examples where the antecedent is not further than ```--contrapro-ctx-size``` parameter.

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_contrapro \
    --results-dir . \
    --results-suffix 'sentence' \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 0 \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```

### Sentence-level NLLB-600M

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_contrapro   \
    --results-dir . \
    --results-suffix 'sentence' \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --limit-plots 0 \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```


### Context-aware OpusMT en-de

The scripts below evaluate the context-aware fine-tuned models. 
Provide the path to the model and the tokenizer through the ```--model-path``` and ```--tokenizer-path``` parameters.

```shell

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_contrapro \
    --results-dir . \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0 \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```

### Context-aware NLLB-600M on ContraPro

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_contrapro \
    --results-dir . \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0 \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```


### Sentence-level NLLB-600M on LCPT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_contrapro \
    --results-dir . \
    --contrapro-dir path/to/repo/data/LCPT \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang fr \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --limit-plots 0  \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```

### Context-aware NLLB-600M on LCPT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_contrapro \
    --results-dir . \
    --contrapro-dir path/to/repo/data/LCPT \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang fr \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0 \
    --max-length 200 \
    --generate-translations \
    --batch-size 500
```


## Translating

The following shows how to translate the datasets using the models.
For each dataset specify the parameters:
- `--dataset` - the dataset name (ContraPro, iwslt2017), use ContraPro for LCPT,
- `--base-data-dir` - the directory where the processed data is stored (e.g., `data/ContraPro_trans_hf`),
- `--raw-data-dir` - the directory where the raw data is stored (e.g., `data/ContraPro`) this is not required for IWSLT as the data will be downloaded from the huggingface repository.
Below are some examples.
- 
### Context-aware OpusMT en-de on ContraPro

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_translate \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --model-name opus_mt \
    --results-dir . \
    --results-suffix checkpoint_best \
    --dataset ContraPro \
    --base-data-dir path/to/repo/data/ContraPro_trans_hf \
    --raw-data-dir path/to/repo/data/ContraPro \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --max-length 200
```

### Sentence-level NLLB-600M on LCPT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_translate \
--model-path "facebook/nllb-200-distilled-600M" \
--tokenizer-path "facebook/nllb-200-distilled-600M" \
--model-name nllb \
--results-dir . \
--dataset ContraPro \
--base-data-dir path/to/repo/data/LCPT_trans_hf \
--raw-data-dir path/to/repo/data/LCPT \
--src-lang en \
--tgt-lang fr \
--src-ctx-size 1 \
--tgt-ctx-size 1 \
--max-length 200
```

### Sentence-level NLLB-600M on IWSLT 2017 en-de validation split

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_translate \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --model-name nllb \
    --results-dir . \
    --results-suffix "en-de.valid" \
    --dataset iwslt2017 \
    --base-data-dir path/to/repo/data/iwslt2017_hf \
    --dataset-splits "valid" \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --max-length 200
```

## Disabling Heads

The script below disables all heads of the model and saves the results to the `disabled_heads_results.tsv` file.

### Sentence-level Opus MT on sentence-level dataset

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 0 \
    --disable-all-model-heads \
    --batch-size 500
```

### Sentence-level NLLB-600M on sentence-level (filtered ctx-0) datset

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --limit-plots 0 \
    --disable-all-model-heads \
    --batch-size 500
```

### Context-aware Opus MT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0 \
    --disable-all-model-heads \
    --batch-size 500
```

#### Context-aware NLLB-600M LCPT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_disable_heads_contrapro \
    --results-dir . \
    --save-results-file disabled_heads_results.tsv \
    --contrapro-dir path/to/repo/data/LCPT \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang fr \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0  \
    --max-length 200 \
    --disable-all-model-heads \
    --batch-size 500
```


## Modifying heads

The following shows how to modify the attention heads of the models while evaluating on ContraPro or LCPT.

Parameter `--modify-heads-simultaneously` is used to modify all heads simultaneously. 
Otherwise, each head will be modified separately.

Specify the list of values to modify the heads to using the `--modify-heads-to-values` parameter.

### Modifying specific heads

Here the script will modify the specified heads of the OpusMT en-de model to the specified values.
Parameter `--modify-heads-simultaneously` is used to modify all heads simultaneously. 
Otherwise, each head will be modified separately.

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 1 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --model-path "Helsinki-NLP/opus-mt-en-de" \
    --tokenizer-path "Helsinki-NLP/opus-mt-en-de" \
    --limit-plots 0 \
    --modify-heads "encoder,5,0" "phrase_cross,5,0" "context_cross,4,0" "decoder_after,5,3" "decoder_after,5,5" \
    --modify-heads-to-values 0.01 0.99 \
    --modify-heads-simultaneously
```

### Modify all heads of the model

#### Sentence-level Opus MT en-de

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir path/to/repo/data/ContraPro \
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
    --modify-heads-to-values 0.01 0.25 0.5 0.75 0.99
```


#### Context-aware Opus MT en-de with context size 3

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-ctx-size 3 \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 3 \
    --tgt-ctx-size 3 \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --limit-plots 0 \
    --modify-all-model-heads \
    --batch-size 500 \
    --modify-heads-to-values 0.01 0.25 0.5 0.75 0.99
```

#### Sentence-level NLLB-200 on LCPT

```shell
PYTHONPATH=path/to/repo/src/ python -m evaluating.m2m_100_modify_attention_contrapro \
    --results-dir . \
    --save-results-file modified_heads_results.tsv \
    --contrapro-dir path/to/repo/data/LCPT \
    --contrapro-ctx-size 0 \
    --filter-context-size \
    --src-lang en \
    --tgt-lang fr \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --limit-plots 0 \
    --modify-all-model-heads \
    --batch-size 500 \
    --modify-heads-to-values 0.01 0.99

```

## Fine-tune Heads

The following shows how to fine-tune the attention heads of the models on the ContraPro dataset.
We only provide the script for training the OpusMT model.
Use ```--tuned-heads``` to specify the heads to fine-tune.

The script will also evaluate the model on the ContraPro dataset after training. 
Additionally, it will translate the test split of the IWSLT 2017 dataset.

### Sentence-level Opus MT

```shell
PYTHONPATH=path/to/repo/src/ python -m head_tuning.opus_mt_head_tuning \
    --model-path "facebook/nllb-200-distilled-600M" \
    --tokenizer-path "facebook/nllb-200-distilled-600M" \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 0 \
    --tgt-ctx-size 0 \
    --dataset ctxpro \
    --base-dataset-dir path/to/repo/data/ctxpro_iwslt2017ende \
    --processed-dataset-dir path/to/repo/data/ctxpro_hf/opus_mt_ctx_0 \
    --test-size 0.0 \
    --split-seed 1 \
    --tuned-heads "context_cross,4,7" \
    --optimizer adafactor \
    --tuning-loss mse \
    --lambda-prediction 0.0 \
    --lambda-head-tune 1.0 \
    --lambda-head-stabilize 0.0 \
    --freeze-non-qk \
    --max-length  200 \
    --max-token-idx-length 20 \
    --learning-rate 1e-3 \
    --weight-decay 1e-2 \
    --warmup-ratio 0.0 \
    --per-device-train-batch-size 12 \
    --per-device-eval-batch-size 12 \
    --gradient-accumulation-steps 8 \
    --use-fp16 \
    --num-train-epochs 10 \
    --save-total-limit 1 \
    --contrapro-after-training \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-batch-size 500 \
    --translate-after-training \
    --translate-dataset-name iwslt2017 \
    --translate-dataset-splits test \
    --translate-base-data-dir path/to/repo/data/iwslt2017_hf \
    --wandb-project attention_anaysis

```

### Context-aware Opus MT

```shell
PYTHONPATH=path/to/repo/src/ python -m head_tuning.opus_mt_head_tuning \
    --model-path path/to/fine_tuned_model/model/ \
    --tokenizer-path path/to/fine_tuned_model/tokenizer/ \
    --src-lang en \
    --tgt-lang de \
    --src-ctx-size 1 \
    --tgt-ctx-size 1 \
    --dataset ctxpro \
    --base-dataset-dir path/to/repo/data/ctxpro_iwslt2017ende \
    --processed-dataset-dir path/to/repo/data/ctxpro_hf/opus_mt_ctx_1 \
    --test-size 0.0 \
    --split-seed 1 \
    --tuned-heads "encoder,1,7" \
    --optimizer adafactor \
    --tuning-loss mse \
    --lambda-prediction 0.0 \
    --lambda-head-tune 1.0 \
    --lambda-head-stabilize 0.0 \
    --freeze-non-qk \
    --max-length  200 \
    --max-token-idx-length 20 \
    --learning-rate 1e-3 \
    --weight-decay 1e-2 \
    --warmup-ratio 0.0 \
    --per-device-train-batch-size 12 \
    --per-device-eval-batch-size 12 \
    --gradient-accumulation-steps 8 \
    --use-fp16 \
    --num-train-epochs 10 \
    --save-total-limit 1 \
    --contrapro-after-training \
    --contrapro-dir path/to/repo/data/ContraPro \
    --contrapro-batch-size 200 \
    --translate-after-training \
    --translate-dataset-name iwslt2017 \
    --translate-dataset-splits test \
    --translate-base-data-dir path/to/repo/data/iwslt2017_hf \
    --wandb-project attention_anaysis
```

## AI Assistant Usage

During writing the code in this repository we used Github Copilot (https://github.com/features/copilot).