from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple
import warnings

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

# from transformers.trainer import is_sagemaker_mp_enabled, smp_forward_backward, OptimizerNames

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction

from dataclasses import field
from pathlib import Path

from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import add_start_docstrings

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from transformers import (
    PreTrainedTokenizerBase
)

from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

# from attention_transfer_hf.transfer_trainer import TransferSeq2SeqTrainer, TransferSeq2SeqTrainingArguments
from transformers.utils import PaddingStrategy

# TOKEN_TYPES = (
#     "antecedent",
#     "pronoun",
# )

# attention_types in ("encoder", "context_cross", "phrase_cross", "decoder", "decoder_after")
ATTENTION_TYPES = {
    "encoder": "encoder_attentions",
    "context_cross": "cross_attentions",
    "phrase_cross": "cross_attentions",
    "decoder": "decoder_attentions",
    "decoder_after": "decoder_attentions",
}

PRE_ATTENTION_TYPES = {
    "encoder": "encoder_pre_attentions",
    "context_cross": "cross_pre_attentions",
    "phrase_cross": "cross_pre_attentions",
    "decoder": "decoder_pre_attentions",
    "decoder_after": "decoder_pre_attentions",
}


@dataclass
@add_start_docstrings(Seq2SeqTrainingArguments.__doc__)
class HeadTuneSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    # tuned_attention: str = field(
    #     default='encoder', metadata={'help': f'The attention to be tuned {tuple(ATTENTION_TYPES.keys())}'}
    # )
    # tuned_token_type: str = field(
    #     default='src_antecedent', metadata={'help': f'The token type to be tuned {TOKEN_TYPES}'}
    # )
    tuned_heads: list = field(
        default=None, metadata={
            'help': 'a list of tuples of heads to be tuned in the form of (attention_type, layer, head), '
                    'where attention_type in ("encoder", "context_cross", "phrase_cross", "decoder", "decoder_after")'
        })
    # tuned_head_layer: int = field(
    #     default=0, metadata={"help": "The layer of the head to tune."})
    # tuned_head_index: int = field(
    #     default=0, metadata={"help": "The index of the head to tune."})
    lambda_prediction: float = field(
        default=1.0, metadata={"help": "The coefficient of the prediction loss."})
    lambda_head_tune: float = field(
        default=1.0, metadata={"help": "The coefficient of the head tuning loss."})
    lambda_head_stabilize: float = field(
        default=1.0, metadata={"help": "The coefficient of the head stabilizing loss."})
    # use_pre_attentions: bool = field(
    #     default=False, metadata={'help': f'If true use pre-softmax attentions for the calculation of loss.'}
    # )
    tune_loss_type: str = field(
        default='mse', metadata={'help': 'The loss type for tuning heads ("mse", "kl", "mse_post).'}
    )
    ignore_index: int = field(
        default=-100, metadata={'help': 'The index to ignore in the loss calculation.'}
    )


class HeadTuneSeq2SeqTrainer(Seq2SeqTrainer):

    def __init__(
            self,
            model: Union["PreTrainedModel", nn.Module] = None,
            teacher: Union["PreTrainedModel", nn.Module] = None,
            args: "HeadTuneSeq2SeqTrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.teacher = teacher
        # self.tuned_attention = args.tuned_attention
        # self.tuned_token_type = args.tuned_token_type
        self.tuned_heads = args.tuned_heads
        # self.tuned_head_layer = args.tuned_head_layer
        # self.tuned_head_index = args.tuned_head_index
        self.lambda_prediction = args.lambda_prediction
        self.lambda_head_tune = args.lambda_head_tune
        self.lambda_head_stabilize = args.lambda_head_stabilize
        self.tune_loss_type = args.tune_loss_type
        self.use_pre_attentions = self.tune_loss_type == 'mse'
        self.ignore_index = args.ignore_index

        if self.tune_loss_type == 'mse':
            self.loss_fn = self.calculate_mse_loss
        elif self.tune_loss_type == 'mse_post':
            self.loss_fn = self.calculate_mse_loss
        elif self.tune_loss_type == 'kl':
            self.loss_fn = self.calculate_kl_loss
        else:
            warnings.warn(f"Unknown loss type '{self.tune_loss_type}'. Using mse loss.")
            self.loss_fn = self.calculate_mse_loss
            self.use_pre_attentions = True

        print(f'Using {self.tune_loss_type} loss (pre-softmax: {self.use_pre_attentions}) for head tuning.')

        assert all([attention_type in ATTENTION_TYPES for attention_type, _, _ in self.tuned_heads])

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.teacher.generation_config = gen_config

        if self.place_model_on_device and not getattr(self.teacher, "is_loaded_in_8bit", False):
            self._move_model_to_device(self.teacher, args.device)

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            return deepcopy(gen_config_arg)

        # str or Path
        pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
        config_file_name = None

        # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
        # This step is required in order to determine config_file_name
        if pretrained_model_name.is_file():
            config_file_name = pretrained_model_name.name
            pretrained_model_name = pretrained_model_name.parent
        # dir path
        elif pretrained_model_name.is_dir():
            pass
        # model id or URL
        else:
            pretrained_model_name = gen_config_arg

        gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
        return gen_config

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.
    #
    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #         return loss_mb.reduce_mean().detach().to(self.args.device)
    #
    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)
    #
    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #
    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)
    #
    #     return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        input_mask = inputs['attention_mask']
        if 'labels' in inputs:
            output_mask = 1 - (inputs['labels'] == self.ignore_index).int()

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # 'src_context_idx', 'src_phrase_idx', 'tgt_context_idx', 'tgt_phrase_idx'
        src_context_idx = inputs.pop("src_context_idx") if "src_context_idx" in inputs else None
        src_phrase_idx = inputs.pop("src_phrase_idx") if "src_phrase_idx" in inputs else None
        tgt_context_idx = inputs.pop("tgt_context_idx") if "tgt_context_idx" in inputs else None
        tgt_phrase_idx = inputs.pop("tgt_phrase_idx") if "tgt_phrase_idx" in inputs else None

        # outputs = model(**inputs, output_attentions=True)
        outputs = self.model(**inputs, output_attentions=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_attentions=True)

        tune_losses = []
        stabilize_losses = []
        # total_tune_loss = None
        # total_stabilize_loss = None
        for attention_type, layer, head in self.tuned_heads:
            tune_loss, stabilize_loss = self.calculate_knowledge_transfer_loss(
                student_outputs=outputs,
                teacher_outputs=teacher_outputs,
                input_mask=input_mask,
                output_mask=output_mask,
                tuned_attention=attention_type,
                tuned_head_layer=layer,
                tuned_head_index=head,
                src_context_idx=src_context_idx,
                src_phrase_idx=src_phrase_idx,
                tgt_context_idx=tgt_context_idx,
                tgt_phrase_idx=tgt_phrase_idx,
            )
            tune_losses.append(tune_loss)
            stabilize_losses.append(stabilize_loss)

        total_tune_loss = torch.sum(
            torch.stack(tune_losses)
        )
        total_stabilize_loss = torch.sum(
            torch.stack(stabilize_losses)
        )

        loss = (self.lambda_prediction * loss
                + self.lambda_head_tune * total_tune_loss
                + self.lambda_head_stabilize * total_stabilize_loss)

        return (loss, outputs) if return_outputs else loss

    def create_modification_mask(self, device, batch_size, num_src_tokens, num_tgt_tokens,
                                 modify_src_tokens, modify_tgt_tokens):
        head_modification_mask = torch.zeros(
            (batch_size, num_src_tokens, num_tgt_tokens),
            dtype=torch.float, device=device)

        if modify_src_tokens is not None and modify_tgt_tokens is not None:
            for bs in range(batch_size):
                for src_tokens_i in range(modify_src_tokens[bs].shape[0]):
                    src_index = modify_src_tokens[bs, src_tokens_i]
                    if src_index < 0:
                        continue
                    for tgt_tokens_i in range(modify_tgt_tokens[bs].shape[0]):
                        tgt_index = modify_tgt_tokens[bs, tgt_tokens_i]
                        if tgt_index < 0:
                            continue
                        head_modification_mask[bs, src_index, tgt_index] = 1

        return head_modification_mask

    def create_target_mask(self, device, batch_size, num_src_tokens, num_tgt_tokens,
                           modify_src_tokens):
        target_mask = torch.zeros(
            (batch_size, num_src_tokens, num_tgt_tokens),
            dtype=torch.float, device=device)

        if modify_src_tokens is not None:
            # for bs in range(batch_size):
            #     for tgt_tokens_i in range(modify_tgt_tokens[bs].shape[0]):
            #         tgt_index = modify_tgt_tokens[bs, tgt_tokens_i]
            #         if tgt_index < 0:
            #             continue
            #         target_mask[bs, :, tgt_index] = 1
            for bs in range(batch_size):
                for src_tokens_i in range(modify_src_tokens[bs].shape[0]):
                    src_index = modify_src_tokens[bs, src_tokens_i]
                    if src_index < 0:
                        continue
                    target_mask[bs, src_index, :] = 1

        return target_mask

    def calculate_mse_loss(self, student_attention, target_attention, mask):
        """
        Calculate the mean squared error loss between student and teacher attentions.
        :param student_attention:
        :param target_attention:
        :param mask:
        :return:
        """
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, dtype=student_attention.dtype, device=student_attention.device)

        diff = (student_attention - target_attention) * mask
        diff = torch.clamp(diff, -1e10, 1e10)
        se = diff ** 2
        masked_loss = torch.where(mask.bool(), se, torch.tensor(torch.nan, dtype=se.dtype, device=se.device))
        loss = masked_loss.nanmean()
        return loss

    def calculate_kl_loss(self, student_attention, target_attention, mask):
        """
        Calculate the KL divergence loss between student and teacher attentions.
        :param student_attention:
        :param target_attention:
        :param mask:
        :return:
        """
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, dtype=student_attention.dtype, device=student_attention.device)

        loss_pointwise = target_attention * (target_attention.log() - student_attention.log())
        loss = torch.sum(torch.nan_to_num(loss_pointwise) * mask) / torch.sum(mask)
        return loss

    def calculate_knowledge_transfer_loss(self, student_outputs, teacher_outputs,
                                          input_mask, output_mask,
                                          tuned_attention,
                                          tuned_head_layer, tuned_head_index,
                                          src_context_idx,
                                          src_phrase_idx,
                                          tgt_context_idx,
                                          tgt_phrase_idx, ):
        src_token_idx = None
        tgt_token_idx = None
        causal_mask = None
        # tuned_attention in ("encoder", "phrase_cross", "context_cross", "decoder", "decoder_after")
        if tuned_attention == 'encoder':
            src_token_idx = src_phrase_idx
            tgt_token_idx = src_context_idx
            src_mask = input_mask
            tgt_mask = input_mask
        elif tuned_attention == 'phrase_cross':
            src_token_idx = tgt_phrase_idx
            tgt_token_idx = src_phrase_idx
            src_mask = output_mask
            tgt_mask = input_mask
        elif tuned_attention == 'context_cross':
            src_token_idx = tgt_phrase_idx
            tgt_token_idx = src_context_idx
            src_mask = output_mask
            tgt_mask = input_mask
        elif tuned_attention == 'decoder':
            src_token_idx = tgt_phrase_idx
            tgt_token_idx = tgt_context_idx
            src_mask = output_mask
            tgt_mask = output_mask
            causal_mask = 1 - torch.triu(torch.ones((output_mask.shape[-1], output_mask.shape[-1]),
                                                    device=output_mask.device), diagonal=1)
        elif tuned_attention == 'decoder_after':
            src_token_idx = tgt_phrase_idx
            tgt_token_idx = None
            # if tgt_context_idx is not None:
            tgt_context_padding_mask = (tgt_context_idx == -1).to(tgt_context_idx.dtype)
            tgt_token_idx = (tgt_context_idx + 1) * (1 - tgt_context_padding_mask) \
                            + tgt_context_idx * tgt_context_padding_mask
            src_mask = output_mask
            tgt_mask = output_mask
            causal_mask = 1 - torch.triu(torch.ones((output_mask.shape[-1], output_mask.shape[-1]),
                                                    device=output_mask.device), diagonal=0)

        if self.use_pre_attentions:
            student_attention = student_outputs[PRE_ATTENTION_TYPES[tuned_attention]]
            teacher_attention = teacher_outputs[PRE_ATTENTION_TYPES[tuned_attention]]
        else:
            student_attention = student_outputs[ATTENTION_TYPES[tuned_attention]]
            teacher_attention = teacher_outputs[ATTENTION_TYPES[tuned_attention]]

        student_attention = student_attention[tuned_head_layer][:, tuned_head_index, ...]
        teacher_attention = teacher_attention[tuned_head_layer][:, tuned_head_index, ...]
        device = teacher_attention.device
        shape = teacher_attention.shape
        head_disturbance_mask = self.create_modification_mask(device, shape[0], shape[1], shape[2],
                                                              src_token_idx, tgt_token_idx)

        # target_mask = self.create_target_mask(device, shape[0], shape[1], shape[2], src_token_idx)
        target_mask = head_disturbance_mask.max(dim=-1, keepdim=True)[0].broadcast_to(head_disturbance_mask.shape)
        rev_target_mask = 1 - target_mask
        target_mask = target_mask * src_mask[..., None] * tgt_mask[..., None, :]
        rev_target_mask = rev_target_mask * src_mask[..., None] * tgt_mask[..., None, :]

        if causal_mask is not None:
            target_mask = target_mask * causal_mask[None, ...]
            rev_target_mask = rev_target_mask * causal_mask[None, ...]

        if self.use_pre_attentions:
            rev_disturbance_mask = 1 - head_disturbance_mask

            n = torch.maximum(torch.sum(head_disturbance_mask, dim=-1), torch.tensor(1.0, device=device))
            b = 0.99

            teacher_pre_attention = teacher_attention if self.use_pre_attentions else torch.log(teacher_attention)
            a = torch.sum((torch.exp(teacher_pre_attention) * rev_disturbance_mask), dim=-1)
            y = a * b / ((1 - b) * n)
            x = torch.unsqueeze(torch.log(y), dim=-1)
            teacher_pre_attention = teacher_pre_attention * rev_disturbance_mask + x * head_disturbance_mask

            target_attention = teacher_pre_attention
        else:
            token_mask = head_disturbance_mask.max(dim=-1, keepdim=True)[0]
            target_attention = teacher_attention * (1 - token_mask) + head_disturbance_mask

        tune_loss = self.loss_fn(student_attention, target_attention, target_mask)
        stabilize_loss = self.loss_fn(student_attention, target_attention, rev_target_mask)

        return tune_loss, stabilize_loss


class HeadTuneGenerationConfig(GenerationConfig):
    def __init__(self, ignore_model_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_model_kwargs = ignore_model_kwargs if ignore_model_kwargs is not None else []

    def update(self, **kwargs):
        unused_kwargs = super().update(**kwargs)
        unused_kwargs = {key: value for key, value in unused_kwargs.items() if key not in self.ignore_model_kwargs}
        return unused_kwargs
