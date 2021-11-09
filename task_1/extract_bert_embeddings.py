# Features: BERT Embeddings
# Models: SVM

import sys
import os

import json
import re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from definitions import PROC_DATA_PATH, BERT_EMB_PATH

from helper_funcs import get_word_sent_embedding, device


def extract_bert_embeddings(dataset: str = 'covid_tweets'):
    data_path = PROC_DATA_PATH

    dsplits = ['train', 'val', 'test']
    split_data = []
    for split_type in dsplits:
        with open(
            os.path.join(data_path, f'{dataset}_{split_type}_data.json'),
            'r',
            encoding='utf-8',
        ) as f:
            split_data.append(json.load(f))

    # train_data, val_data, test_data = data
    import numpy as np
    bert_list = ['bert-base-uncased', 'bert-large-uncased']
    for bert_type in bert_list:
        print(bert_type)
        tokenizer = BertTokenizer.from_pretrained(bert_type)
        model = BertModel.from_pretrained(
            bert_type,
            output_hidden_states=True,
            return_dict=False,
        )
        model.to(device).eval()

        categories = [
            'sent_word_catavg', 'sent_word_catavg_wostop', 'sent_word_sumavg',
            'sent_word_sumavg_wostop', 'sent_emb_2_last', 'sent_emb_2_last_wostop',
            'sent_emb_last', 'sent_emb_last_wostop', 'labels'
        ]

        embed_dict = {
            'train': {c: [] for c in categories},
            'val': {c: [] for c in categories},
            'test': {c: [] for c in categories[:-1]}
        }
        dataframes = {'sent_word_catavg': []}

        # data_type = {'train': train_data, 'val': val_data, 'test': test_data}

        data_type = {k: v for k, v in list(zip(dsplits, split_data))}

        for phase in tqdm(data_type):
            data = data_type[phase]
            # ids = list(data.keys())
            # print(data.keys())
            # break
            for i, id in tqdm(enumerate(data)):

                text = data[id]['text']
                marked_text = "[CLS] "+text+" [SEP]"

                sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
                    sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop = get_word_sent_embedding(
                        marked_text, model, tokenizer)

                y = get_word_sent_embedding(marked_text, model, tokenizer)
                print(type(y))
                # print(type(sent_word_catavg), type(sent_word_catavg[0]))
                # print(dataframes['sent_word_catavg'])
                # qq =
                # dataframes['sent_word_catavg'] = np.r_[int(id), sent_word_catavg] \
                #     if 'sent_word_catavg' not in dataframes \
                #     else np.append(
                #     dataframes['sent_word_catavg'],
                #     np.r_[int(id), sent_word_catavg],
                #     axis=0
                # )
                dataframes['sent_word_catavg'].append(
                    [int(id), *sent_word_catavg])
                print(pd.DataFrame(dataframes['sent_word_catavg']))
                # print(len(sent_word_catavg), len(sent_word_catavg_wostop))
                # print(len(sent_word_sumavg), len(sent_word_sumavg_wostop))
                # print(len(sent_), len(sent_word_sumavg_wostop))
                if i == 2:
                    exit()
                embed_dict[phase]['sent_word_catavg'].append(
                    sent_word_catavg.tolist())
                embed_dict[phase]['sent_word_sumavg'].append(
                    sent_word_sumavg.tolist())
                embed_dict[phase]['sent_emb_2_last'].append(
                    sent_emb_2_last.tolist())
                embed_dict[phase]['sent_emb_last'].append(
                    sent_emb_last.tolist())
                embed_dict[phase]['sent_word_catavg_wostop'].append(
                    sent_word_catavg_wostop.tolist())
                embed_dict[phase]['sent_word_sumavg_wostop'].append(
                    sent_word_sumavg_wostop.tolist())
                embed_dict[phase]['sent_emb_2_last_wostop'].append(
                    sent_emb_2_last_wostop.tolist())
                embed_dict[phase]['sent_emb_last_wostop'].append(
                    sent_emb_last_wostop.tolist())
                if phase != 'test':
                    embed_dict[phase]['labels'].append(data[id]['label'])

        # json.dump(embed_dict, open(
        #     data_path+'/bert_embs/%s_raw_text.json' % (bert_type), 'w'))
        with open(
            os.path.join(
                BERT_EMB_PATH,
                f'{dataset}_{bert_type}_raw_text.json'
            ),
            'w',
        ) as rawf:
            json.dump(embed_dict, rawf)

        # Bert Embeddings on processed text
        embed_dict = {
            'train': {c: [] for c in categories},
            'val': {c: [] for c in categories},
            'test': {c: [] for c in categories[:-1]}
        }

        for phase in data_type:
            data = data_type[phase]
            for id in data:
                proc_text = data[id]['processed']

                proc_text = [word for word in proc_text if not re.search(
                    r'<(/?)[a-z]+>', word)]

                text = ""
                for word in proc_text:
                    text += word if word in [',', '.'] else " "+word

                marked_text = "[CLS] "+text+" [SEP]"

                sent_word_catavg, sent_word_catavg_wostop, sent_word_sumavg, sent_word_sumavg_wostop, \
                    sent_emb_2_last, sent_emb_2_last_wostop, sent_emb_last, sent_emb_last_wostop = get_word_sent_embedding(
                        marked_text, model, tokenizer)

                embed_dict[phase]['sent_word_catavg'].append(
                    sent_word_catavg.tolist())
                embed_dict[phase]['sent_word_sumavg'].append(
                    sent_word_sumavg.tolist())
                embed_dict[phase]['sent_emb_2_last'].append(
                    sent_emb_2_last.tolist())
                embed_dict[phase]['sent_emb_last'].append(
                    sent_emb_last.tolist())
                embed_dict[phase]['sent_word_catavg_wostop'].append(
                    sent_word_catavg_wostop.tolist())
                embed_dict[phase]['sent_word_sumavg_wostop'].append(
                    sent_word_sumavg_wostop.tolist())
                embed_dict[phase]['sent_emb_2_last_wostop'].append(
                    sent_emb_2_last_wostop.tolist())
                embed_dict[phase]['sent_emb_last_wostop'].append(
                    sent_emb_last_wostop.tolist())
                if phase != 'test':
                    embed_dict[phase]['labels'].append(data[id]['label'])

        # json.dump(embed_dict, open(
        #     data_path+'/bert_embs/%s_proc_text.json' % (bert_type), 'w'))
        with open(
            os.path.join(
                BERT_EMB_PATH,
                f'{dataset}_{bert_type}_proc_text.json'
            ),
            'w',
        ) as rawf:
            json.dump(embed_dict, rawf)


extract_bert_embeddings(dataset='political_debates')
