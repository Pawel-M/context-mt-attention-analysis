# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import json
import os

import datasets

_DESCRIPTION = """"""
_HOMEPAGE_URL = ""
_CITATION = """"""

_VERSION = "0.0.1"
_BASE_URL = ""

# Please note that only few pairs are shown here. You can use config to generate data for all language pairs
_LANGUAGE_PAIRS = [
    ("en", "de"),
]


class ContraProConfig(datasets.BuilderConfig):
    def __init__(self, *args, ctx_size=0, files_base_name='contrapro', tgt_phrase_key='ref pronoun', **kwargs):
        super().__init__(
            *args,
            name=f"{ctx_size}",
            **kwargs,
        )
        self.lang1 = 'en'
        self.lang2 = 'de'
        self.ctx_size = ctx_size
        self.files_base_name = files_base_name
        self.tgt_phrase_key = tgt_phrase_key


class ContraPro(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ContraProConfig(
            description=f"Translating {lang1} to {lang2} or vice versa",
            version=datasets.Version(_VERSION),
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = ContraProConfig

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.config.ctx_size = kwargs.get('ctx_size', 0)
    #     self.config.files_base_name = kwargs.get('files_base_name', 'contrapro')
    #     self.config.tgt_phrase_key = kwargs.get('tgt_phrase_key', 'ref pronoun')

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document": datasets.Value("string"),
                    "translation": datasets.Translation(languages=(self.config.lang1, self.config.lang2)),
                    "context_phrase": {
                        self.config.lang1: datasets.Value("string"),
                        self.config.lang2: datasets.Value("string"),
                    },
                    "phrase": {
                        self.config.lang1: datasets.Value("string"),
                        self.config.lang2: datasets.Value("string"),
                    },
                    "context": {
                        self.config.lang1: datasets.Sequence(datasets.Value("string")),
                        self.config.lang2: datasets.Sequence(datasets.Value("string")),
                    },
                    "context_distance": datasets.Value("int32"),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        download_url = _BASE_URL.format(self.config.lang1, self.config.lang2)
        # path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": self.base_path},
            )
        ]

    # @classmethod
    # def _extract_info(cls, sentence_id):
    #     # see https://github.com/huggingface/datasets/issues/1844
    #     # sentence ids have the following format: en/2017/7006210/7050201.xml.gz
    #     # lang/year/imdb_id/opensubtitles_id.xml.gz
    #     parts = sentence_id[: -len(".xml.gz")].split("/")
    #     parts.pop(0)  # remove lang, we do not need it
    #
    #     # returns year, imdb_id, opensubtitles_id
    #     return tuple(map(int, parts))

    def load_contrapro(self, dir='../../Datasets/ContraPro', files_base_name='contrapro'):
        print(f'Loading ContraPro from {dir}')
        contrapro_file = os.path.join(dir, f'{files_base_name}.json')
        with open(contrapro_file, 'r') as f:
            data_json = json.load(f)

        data = []
        for d in data_json:
            if d['ante distance'] != 0:
                continue

            tgts = [d['ref segment']]
            tgts.extend([e['contrastive'] for e in d['errors']])

            data.append((d['src segment'], tgts, d))
        return data

    def load_contrapro_with_context(self, context_size=1, dir='../../Datasets/ContraPro', files_base_name='contrapro'):
        print(f'Loading ContraPro with context from {dir}')
        working_context_size = context_size if context_size is not None else 2

        contrapro_file = os.path.join(dir, f'{files_base_name}.json')
        context_dir = os.path.join(dir, f'ctx{working_context_size}')
        src_context_file = os.path.join(context_dir, f'{files_base_name}.context.en')
        tgt_context_file = os.path.join(context_dir, f'{files_base_name}.context.de')

        with open(contrapro_file, 'r') as f:
            data_json = json.load(f)

        with open(src_context_file, 'r') as f:
            src_context_lines = f.readlines()

        with open(tgt_context_file, 'r') as f:
            tgt_context_lines = f.readlines()

        print(f'Processing {len(data_json)} examples from {contrapro_file}...')

        data = []
        context_id = 0
        for d in data_json:
            tgts = [d['ref segment']]
            contrastive = [e['contrastive'] for e in d['errors']]
            tgts.extend(contrastive)

            context_start_line = context_id * working_context_size
            src_context = src_context_lines[context_start_line:context_start_line + working_context_size]
            tgt_context = tgt_context_lines[context_start_line:context_start_line + working_context_size]

            # remove newline symbols
            src_context = [c[:-1] for c in src_context]
            tgt_context = [c[:-1] for c in tgt_context]
            context_id += len(tgts)
            if context_size is not None and d['ante distance'] > context_size:
                continue
            data.append((d['src segment'], tgts, src_context, tgt_context, d))
        return data

    def _generate_examples(self, datapath):
        l1, l2 = self.config.lang1, self.config.lang2
        ctx_size = self.config.ctx_size
        files_base_name = self.config.files_base_name
        tgt_phrase_key = self.config.tgt_phrase_key
        if ctx_size > 0:
            data = self.load_contrapro_with_context(ctx_size, datapath, files_base_name)
        else:
            data = self.load_contrapro(datapath, files_base_name)

        # previous_l1_id, previous_l2_id = None, None
        for sentence_counter, data_point in enumerate(data):
            d = data_point[-1]

            if len(data_point) > 3:
                src_context = data_point[2]
                tgt_context = data_point[3]
            else:
                src_context = None
                tgt_context = None

            document_id = d['document id']
            source = d['src segment'].strip()
            target = d['ref segment'].strip()
            source_antecedent = d['src ante head'].strip()
            source_pronoun = d['src pronoun'].strip()
            target_antecedent = d['ref ante head'].strip() \
                if 'ref ante head' in d and d['ref ante head'] is not None \
                else None
            target_pronoun = d[tgt_phrase_key].strip()
            # wrong_targets = [error['contrastive'].strip() for error in d['errors']]
            # wrong_targets_pronouns = [error['replacement'].strip() for error in d['errors']]
            ante_distance = d['ante distance']

            result = (
                sentence_counter,
                {
                    "id": str(sentence_counter),
                    "document": document_id,
                    "translation": {
                        l1: source,
                        l2: target
                    },
                    "context": {
                        l1: src_context,
                        l2: tgt_context,
                    },
                    "context_phrase": {
                        l1: source_antecedent,
                        l2: target_antecedent,
                    },
                    "phrase": {
                        l1: source_pronoun,
                        l2: target_pronoun,
                    },
                    "context_distance": ante_distance,
                },
            )
            yield result


if __name__ == '__main__':
    from datasets import load_dataset_builder

    ds_builder = ContraPro('../data/ContraPro/ctx0', base_path='../../Datasets/ContraPro', ctx_size=0)
    ds_builder.download_and_prepare('../data/ContraPro/ctx0')
    dataset = ds_builder.as_dataset('train')
    for i in range(10):
        print(dataset[i])
