import os

import datasets

_DESCRIPTION = """

"""

_LANGUAGE_PAIRS = [
    ("en", "de"),
]


class ContextualDatasetConfig(datasets.BuilderConfig):
    def __init__(self, *args, ctx_size=0, **kwargs):
        super().__init__(
            *args,
            name=f"{ctx_size}",
            **kwargs,
        )
        self.lang1 = 'en'
        self.lang2 = 'de'
        self.ctx_size = ctx_size


class ContextualDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ContextualDatasetConfig(
            description=f"Translating {lang1} to {lang2} or vice versa",
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = ContextualDatasetConfig

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.config.ctx_size = kwargs.get('ctx_size', 0)
    #     self.config.files_base_name = kwargs.get('files_base_name', 'contrapro')
    #     self.config.tgt_phrase_key = kwargs.get('tgt_phrase_key', 'ref pronoun')

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document": datasets.Value("string"),
                    "translation": datasets.Translation(languages=[self.config.lang1, self.config.lang2]),
                    "context": {
                        self.config.lang1: datasets.Sequence(datasets.Value("string")),
                        self.config.lang2: datasets.Sequence(datasets.Value("string")),
                    },
                },
            ),
            supervised_keys=None,
            # homepage=_HOMEPAGE_URL,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # download_url = _BASE_URL.format(self.config.lang1, self.config.lang2)
        # path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": 'train', "datapath": self.base_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": 'test', "datapath": self.base_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": 'valid', "datapath": self.base_path},
            )
        ]

    # def load_contrapro(self, dir='../../Datasets/ContraPro', files_base_name='contrapro'):
    #     print(f'Loading ContraPro from {dir}')
    #     contrapro_file = os.path.join(dir, f'{files_base_name}.json')
    #     with open(contrapro_file, 'r') as f:
    #         data_json = json.load(f)
    #
    #     data = []
    #     for d in data_json:
    #         if d['ante distance'] != 0:
    #             continue
    #
    #         tgts = [d['ref segment']]
    #         tgts.extend([e['contrastive'] for e in d['errors']])
    #
    #         data.append((d['src segment'], tgts, d))
    #     return data

    # def load_contrapro_with_context(self, context_size=1, dir='../../Datasets/ContraPro', files_base_name='contrapro'):
    #     print(f'Loading ContraPro with context from {dir}')
    #     working_context_size = context_size if context_size is not None else 2
    #
    #     contrapro_file = os.path.join(dir, f'{files_base_name}.json')
    #     context_dir = os.path.join(dir, f'ctx{working_context_size}')
    #     src_context_file = os.path.join(context_dir, f'{files_base_name}.context.en')
    #     tgt_context_file = os.path.join(context_dir, f'{files_base_name}.context.de')
    #
    #     with open(contrapro_file, 'r') as f:
    #         data_json = json.load(f)
    #
    #     with open(src_context_file, 'r') as f:
    #         src_context_lines = f.readlines()
    #
    #     with open(tgt_context_file, 'r') as f:
    #         tgt_context_lines = f.readlines()
    #
    #     print(f'Processing {len(data_json)} examples from {contrapro_file}...')
    #
    #     data = []
    #     context_id = 0
    #     for d in data_json:
    #         tgts = [d['ref segment']]
    #         contrastive = [e['contrastive'] for e in d['errors']]
    #         tgts.extend(contrastive)
    #
    #         context_start_line = context_id * working_context_size
    #         src_context = src_context_lines[context_start_line:context_start_line + working_context_size]
    #         tgt_context = tgt_context_lines[context_start_line:context_start_line + working_context_size]
    #
    #         # remove newline symbols
    #         src_context = [c[:-1] for c in src_context]
    #         tgt_context = [c[:-1] for c in tgt_context]
    #         context_id += len(tgts)
    #         if context_size is not None and d['ante distance'] > context_size:
    #             continue
    #         data.append((d['src segment'], tgts, src_context, tgt_context, d))
    #     return data

    @classmethod
    def _reset_contexts(cls, lang1_ctx_size, lang2_ctx_size):
        lang1_ctx = [''] * lang1_ctx_size
        lang2_ctx = [''] * lang2_ctx_size
        return lang1_ctx, lang2_ctx

    def _generate_examples(self, split, datapath):
        l1, l2 = self.config.lang1, self.config.lang2
        ctx_size = self.config.ctx_size

        l1_file = os.path.join(datapath, f'{split}.{l1}-{l2}.{l1}')
        l2_file = os.path.join(datapath, f'{split}.{l1}-{l2}.{l2}')
        docids_file = os.path.join(datapath, f'{split}.{l1}-{l2}.docids')

        with open(l1_file, 'r') as f1, open(l2_file, 'r') as f2, open(docids_file, 'r') as d:
            previous_docid = None
            for i, (l1_line, l2_line, docid) in enumerate(zip(f1, f2, d)):
                l1_line = l1_line.strip()
                l2_line = l2_line.strip()
                docid = docid.strip()

                if docid != previous_docid:
                    lang1_ctx, lang2_ctx = self._reset_contexts(ctx_size, ctx_size)

                result = (
                    i,
                    {
                        "id": str(i),
                        "document": docid,
                        "translation": {
                            l1: l1_line,
                            l2: l2_line,
                        },
                        "context": {
                            l1: lang1_ctx.copy(),
                            l2: lang2_ctx.copy(),
                        },
                    },
                )
                yield result

                previous_docid = docid

                lang1_ctx.append(l1_line)
                lang2_ctx.append(l2_line)
                lang1_ctx = lang1_ctx[1:]
                lang2_ctx = lang2_ctx[1:]


if __name__ == '__main__':
    from datasets import load_dataset_builder
    ctx_size = 2

    ds_builder = ContextualDataset(f'../data/wmt2017_hf/en-de/ctx{ctx_size}', '__main__',
                                   base_path='../../VOXReality/data/wmt17/en-de',
                                   ctx_size=ctx_size)
    ds_builder.download_and_prepare(f'../data/wmt2017_hf/en-de/ctx{ctx_size}')
    dataset = ds_builder.as_dataset()
    for i in range(10):
        print(dataset['train'][i])
