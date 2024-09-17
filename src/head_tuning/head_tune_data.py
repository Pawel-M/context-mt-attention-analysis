import string
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from transformers import (
    PreTrainedTokenizerBase
)
import spacy
from transformers.utils import PaddingStrategy

nlp_models = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm')
}


def _split_sentence(sentence, lang):
    nlp = nlp_models[lang]
    # nlp = de_nlp if lang == 'de' else en_nlp
    doc = nlp(sentence)
    word_starts = [w.idx for w in doc]
    words = [w.text for w in doc]
    return word_starts, words


def _find_token_ids(word, sentence, offset_mapping, word_start=-1, use_byte_string=False):
    punctuation_whitespace = string.punctuation + string.whitespace
    if use_byte_string:
        word = word.encode('utf-8')
        sentence = sentence.encode('utf-8')
        punctuation_whitespace = punctuation_whitespace.encode('utf-8')

    word_len = len(word)
    matches = []
    distances = []
    for i in range(len(sentence) - word_len + 1):
        if sentence[i:i + word_len] == word:
            # is it the whole word?
            if (
                    ((i - 1 < 0) or sentence[i - 1] in punctuation_whitespace)
                    and ((i + word_len + 1 >= len(sentence))
                         or sentence[i + word_len] in punctuation_whitespace)
            ):
                matches.append((i, i + word_len))
                distances.append(abs(i - word_start))

    # didn't find the whole word (probably noise in the dataset), just find the occurrences
    if len(matches) == 0:
        for i in range(len(sentence) - word_len + 1):
            if sentence[i:i + word_len] == word:
                matches.append((i, i + word_len))
                distances.append(abs(i - word_start))

    if word_start >= 0:
        try:
            min_distance = min(distances)
            matches = [matches[i] for i in range(len(distances)) if distances[i] == min_distance]
        except:
            print(distances)

    all_token_ids = []
    for span in matches:
        token_ids = []
        all_token_ids.append(token_ids)
        for i in range(offset_mapping.shape[0]):
            if (span[0] <= offset_mapping[i, 1] - 1 and offset_mapping[i, 0] <= span[1] - 1):
                token_ids.append(i)

    return all_token_ids


def analyse_tokens(tokenizer, source, target, source_antecedent, source_pronoun, target_antecedent, target_pronoun,
                   use_byte_string=True):
    # source_word_starts, source_words = _split_sentence(source, 'en')
    source_tokenized = tokenizer(text=source, return_tensors='pt', return_offsets_mapping=True)
    target_tokenized = tokenizer(text_target=target, return_tensors='pt', return_offsets_mapping=True)

    # if filter_by_word_ids:
    #     source_antecedent_word_id = d['src ante head id']
    #     source_antecedent_word_start = source_word_starts[source_antecedent_word_id - 1]
    # else:
    #     source_antecedent_word_start = None
    # source_antecedent_word_id = d['src ante head id']
    # source_antecedent_word_start = source_word_starts[source_antecedent_word_id - 1]

    source_antecedent_token_ids = _find_token_ids(
        source_antecedent, source, source_tokenized['offset_mapping'][0], use_byte_string=use_byte_string,
    )
    source_antecedent_token_ids = source_antecedent_token_ids[0]

    source_pronoun_token_ids = _find_token_ids(source_pronoun, source, source_tokenized['offset_mapping'][0],
                                               use_byte_string=use_byte_string, )
    source_pronoun_token_ids = [token_ids for token_ids in source_pronoun_token_ids if
                                token_ids != source_antecedent_token_ids]
    source_pronoun_token_ids = source_pronoun_token_ids[-1]
    # print(source)
    # print(source_antecedent)
    # print(source_antecedent_token_ids)
    # print(source_pronoun)
    # print(source_pronoun_token_ids)
    # print()

    if target_antecedent is not None:
        target_antecedent_token_ids = _find_token_ids(target_antecedent, target, target_tokenized['offset_mapping'][0],
                                                      use_byte_string=use_byte_string, )
        target_antecedent_token_ids = target_antecedent_token_ids[0]
    else:
        target_antecedent_token_ids = [-1]

    target_pronoun_token_ids = _find_token_ids(target_pronoun, target, target_tokenized['offset_mapping'][0],
                                               use_byte_string=use_byte_string, )
    target_pronoun_token_ids = [token_ids for token_ids in target_pronoun_token_ids if
                                token_ids != target_antecedent_token_ids]
    target_pronoun_token_ids = target_pronoun_token_ids[-1]

    # print(target)
    # print(target_antecedent)
    # print(target_antecedent_token_ids)
    # print(target_pronoun)
    # print(target_pronoun_token_ids)
    # print()

    return source_antecedent_token_ids, source_pronoun_token_ids, target_antecedent_token_ids, target_pronoun_token_ids

    # if len(source_pronoun_token_ids) < 1:
    #     raise Exception()

    # data_point_analysis = {}
    # data_analysis.append(data_point_analysis)
    # data_point_analysis['source_antecedent_token_ids'] = source_antecedent_token_ids
    # data_point_analysis['source_pronoun_token_ids'] = source_pronoun_token_ids
    # data_point_analysis['target_antecedent_token_ids'] = target_antecedent_token_ids
    # data_point_analysis['target_pronouns_token_ids'] = target_pronoun_token_ids
    # data_point_analysis['target_options'] = target_options


def pad_to_length(inputs_list, pad_value, length):
    padded_list = []
    for inputs in inputs_list:
        if len(inputs) > length:
            raise Exception()

        if len(inputs) == length:
            padded_list.append(inputs)
        else:
            padded_list.append(inputs + [pad_value] * (length - len(inputs)))

    return padded_list


def preprocess_function(
        examples,
        tokenizer,
        tokenize_and_analyze_fn,
        src_lang, tgt_lang,
        max_length, max_token_idx_length,
        source_context_size, target_context_size,
        consider_upper_phrases=False,
        # use_byte_string=True
):
    # datasets.DatasetInfo(
    #     description=_DESCRIPTION,
    #     features=datasets.Features(
    #         {
    #             "id": datasets.Value("string"),
    #             "document": datasets.Value("string"),
    #             "translation": datasets.Translation(languages=(self.config.lang1, self.config.lang2)),
    #             "context_phrase": {
    #                 self.config.lang1: datasets.Value("string"),
    #                 self.config.lang2: datasets.Value("string"),
    #             },
    #             "phrase": {
    #                 self.config.lang1: datasets.Value("string"),
    #                 self.config.lang2: datasets.Value("string"),
    #             },
    #             "context": {
    #                 self.config.lang1: datasets.Sequence(datasets.Value("string")),
    #                 self.config.lang2: datasets.Sequence(datasets.Value("string")),
    #             }
    #             "context_distance": datasets.Value("int32"),
    #         },
    #     ),
    #     supervised_keys=None,
    #     homepage=_HOMEPAGE_URL,
    #     citation=_CITATION,
    # )

    sources = [example[src_lang] for example in examples["translation"]]
    targets = [example[tgt_lang] for example in examples["translation"]]
    sources_context = [example[src_lang] for example in examples["context"]]
    targets_context = [example[tgt_lang] for example in examples["context"]]
    sources_context_phrase = [example[src_lang] for example in examples["context_phrase"]]
    targets_context_phrase = [example[tgt_lang] for example in examples["context_phrase"]]
    sources_phrase = [example[src_lang] for example in examples["phrase"]]
    targets_phrase = [example[tgt_lang] for example in examples["phrase"]]
    context_distances = examples["context_distance"]

    (
        model_inputs,
        sources_phrase_indices,
        sources_context_phrase_indices,
        targets_phrase_indices,
        targets_context_phrase_indices
    ) = tokenize_and_analyze_fn(tokenizer, sources, targets,
                                sources_context, targets_context,
                                source_context_size, target_context_size,
                                sources_phrase, targets_phrase,
                                sources_context_phrase, targets_context_phrase,
                                context_distances, max_length, consider_upper_phrases)

    # select the first context phrase indices
    sources_context_phrase_indices = [context[0] if len(context) > 0 else context
                                      for context in sources_context_phrase_indices]
    targets_context_phrase_indices = [context[0] if len(context) > 0 else context
                                      for context in targets_context_phrase_indices]

    # select the last phrase indices
    sources_phrase_indices = [phrase[-1] if len(phrase) > 0 else phrase
                              for phrase in sources_phrase_indices]
    targets_phrase_indices = [phrase[-1] if len(phrase) > 0 else phrase
                              for phrase in targets_phrase_indices]

    model_inputs['src_context_idx'] = pad_to_length(sources_context_phrase_indices, -1, max_token_idx_length)
    model_inputs['src_phrase_idx'] = pad_to_length(sources_phrase_indices, -1, max_token_idx_length)
    model_inputs['tgt_context_idx'] = pad_to_length(targets_context_phrase_indices, -1, max_token_idx_length)
    model_inputs['tgt_phrase_idx'] = pad_to_length(targets_phrase_indices, -1, max_token_idx_length)
    return model_inputs


@dataclass
class HeadTuneDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
