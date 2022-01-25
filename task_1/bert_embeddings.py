

import enum
import os
import json
from typing import List
from numpy import e

from transformers import BertTokenizer, BertModel
from definitions import PROC_DATA_PATH, EMB_DF_PATHS, EMB_CATEGORIES
from helper_funcs import *
from tqdm import tqdm


def load_bert_embeddings(
    dataset: str, text_type: str,
    bert_type: str, emb_cat: List[str] = EMB_CATEGORIES
):
    prefix_phrase = f'{text_type}_{bert_type}_'

    embeddings = {
        c: pd.read_csv(
            os.path.join(EMB_DF_PATHS[dataset], f'{prefix_phrase}{c}.csv'),
            index_col=False,
            header=None,
        ) for c in tqdm(emb_cat, desc='Loading embeddings')
    }

    for embt in embeddings:
        vsize = embeddings[embt].shape[1] - 2
        vcols = [f'p{i}' for i in range(vsize)]

        embeddings[embt].columns = ['id', 'split_type', *vcols]

    return embeddings


def extract_bert_embeddings(dataset: str = 'covid_tweets'):
    data_path = PROC_DATA_PATH

    dsplits = ['train', 'test']
    if dataset == 'covid_tweets':
        dsplits.append('val')

    split_data = []
    for split_type in dsplits:
        with open(
            os.path.join(data_path, f'{dataset}_{split_type}_data.json'),
            'r',
            encoding='utf-8',
        ) as trainf:
            split_data.append(json.load(trainf))

    bert_list = ['bert-base-uncased', 'bert-large-uncased']
    text_list = ['raw', 'processed']
    categories = EMB_CATEGORIES

    data_type = {k: v for k, v in list(zip(dsplits, split_data))}

    for bert_type in bert_list:
        tokenizer = BertTokenizer.from_pretrained(bert_type)
        model = BertModel.from_pretrained(
            bert_type,
            output_hidden_states=True,
            return_dict=False,
        )

        model.to(device).eval()

        for txt_processing in text_list:

            embed_dataframes = {c: [] for c in categories}
            keys, phase_column = [], []

            for phase, data in tqdm(data_type.items(), desc='Phases'):
                # if phase == 'test':
                #     categories = categories[:-1]
                keys += list(data.keys())

                for id in tqdm(data.keys(), desc='Creating embeddings'):
                    text = data[id]['text']

                    if txt_processing == 'processed':
                        text = data[id][txt_processing]
                        proc_text = [
                            word for word in text if not re.search(
                                r'<(/?)[a-z]+>',
                                word
                            )
                        ]

                        to_join = []
                        for word in proc_text:
                            to_join.append(
                                word if word in {',', '.'} else f' {word}'
                            )

                        text = ''.join(to_join)

                    marked_text = f'[CLS] {text} [SEP]'

                    embs = get_word_sent_embedding(
                        marked_text, model, tokenizer)

                    for j, cat in enumerate(categories[:-1]):
                        embed_dataframes[cat].append(embs[j])

                    if phase != 'test':
                        embed_dataframes['labels'].append(data[id]['label'])
                    else:
                        # append -1 to keep labels as big as other categories
                        # so that there is no misalignment
                        embed_dataframes['labels'].append(-1)

                    phase_column.append(phase)

            base_df = pd.DataFrame(keys).merge(
                pd.DataFrame(phase_column),
                how='left',
                left_index=True,
                right_index=True
            )
            for name, frame in tqdm(
                embed_dataframes.items(),
                desc='Writing dataframes'
            ):
                df = base_df.merge(
                    pd.DataFrame(frame),
                    how='left',
                    left_index=True,
                    right_index=True,
                )

                df.to_csv(
                    os.path.join(
                        EMB_DF_PATHS[dataset],
                        f'{txt_processing}_{bert_type}_{name}.csv'
                    ),
                    header=False,
                    index=False
                )


# extract_bert_embeddings(dataset='covid_tweets')
# print()
# x = load_bert_embeddings(
#     dataset='political_debates',
#     text_type='raw',
#     bert_type='bert-large-uncased',
#     emb_cat=[
#         'labels'
#     ]
# )

# print(x['labels'].iloc[:, 2].describe())
