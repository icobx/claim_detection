import os
import re
import sys
import json
import spacy
import pickle

# from urlextract import URLExtract
from tqdm import tqdm
from nltk.corpus import stopwords
from spacy.language import Language
from helper_funcs import *
from sklearn.model_selection import train_test_split

from definitions import INPUT_DATA_PATHS, PROC_DATA_PATH


def preprocess(dataset: str = 'covid_tweets', spacy_nlp: Language = None):
    data_paths = INPUT_DATA_PATHS[dataset]

    spacy_nlp = spacy.load('en_core_web_lg') \
        if spacy_nlp is None else spacy_nlp

    stopwords_set = set(stopwords.words('english'))

    # TODO: mention getting different txt procerssor for polit. debates
    if dataset == 'covid_tweets':
        text_proc = get_text_processor(word_stats='twitter')
    elif dataset == 'political_debates':
        text_proc = get_text_processor(word_stats='english')

    for dsplit_type, split_dict in tqdm(data_paths.items(), desc='data split'):
        data_dict = {}
        lines = None
        with open(split_dict['filepath'], 'r') as f:
            lines = f.readlines()

        # skip header
        for line in tqdm(lines[1:], desc='lines', leave=False):

            line_split = line.strip().split('\t')
            if dataset == 'covid_tweets':
                # topic, link not used anywhere
                topic, id, link, content = line_split[:4]
                claim, label = '0', '0'

                if dsplit_type != 'test':
                    claim, label = line_split[4:]

            elif dataset == 'political_debates':
                i, id, src, content = line_split[:4]
                claim, label = '-1', '0'

                if dsplit_type != 'test':
                    label = line_split[-1]

            prcsd = text_proc.pre_process_doc(content)
            cleaned = [
                word for word in prcsd if not re.search(
                    '[^a-z0-9\s]+',
                    word,
                )
            ]
            cleaned = [
                word for word in cleaned
                if len(word) > 2
                or word.isnumeric()
            ]

            cleaned_ns = [
                word for word in cleaned if word not in stopwords_set
            ]

            spacied = spacy_nlp(' '.join(cleaned))
            spacied_ns = spacy_nlp(' '.join(cleaned_ns))

            pos, pos_ns = [], []
            ner, ner_ns = [], []

            pos = [
                f'{token.text}_{token.pos_}_{token.tag_}' for token
                in spacied
            ]

            pos_ns = [
                f'{token.text}_{token.pos_}_{token.tag_}' for token
                in spacied_ns
            ]

            ner = [
                {
                    'text': entity.text,
                    'label': entity.label_,
                    'start': entity.start_char,
                    'end': entity.end_char,
                } for entity in spacied.ents
            ]
            ner_ns = [
                {
                    'text': entity.text,
                    'label': entity.label_,
                    'start': entity.start_char,
                    'end': entity.end_char,
                } for entity in spacied_ns.ents
            ]

            data_dict[id] = {
                'text': content,
                'processed': prcsd,
                'cleaned': cleaned,
                'cleaned_ns': cleaned_ns,
                'pos': pos,
                'pos_ns': pos_ns,
                'ner': ner,
                'ner_ns': ner_ns,
                'claim': claim,
                'label': label
            }

        # data stats
        pos_dict, pos_ns_dict = {'1': {}, '0': {}}, {'1': {}, '0': {}}
        ner_dict, ner_ns_dict = {'1': {}, '0': {}}, {'1': {}, '0': {}}

        # counter of POS and NER
        for id, val in tqdm(data_dict.items(), desc='data dict'):
            label = str(val['label'])

            for pos_tag in val['pos']:
                tag = pos_tag.split('_')[1]
                pos_dict[label][tag] = 1 \
                    if tag not in pos_dict[label] \
                    else pos_dict[label][tag] + 1

            for pos_tag in val['pos_ns']:
                tag = pos_tag.split('_')[1]
                pos_ns_dict[label][tag] = 1 \
                    if tag not in pos_ns_dict[label] \
                    else pos_ns_dict[label][tag] + 1

            for ner_tag in val['ner']:
                tag = ner_tag['label']
                ner_dict[label][tag] = 1 \
                    if tag not in ner_dict[label] \
                    else ner_dict[label][tag] + 1

            for ner_tag in val['ner_ns']:
                tag = ner_tag['label']
                ner_ns_dict[label][tag] = 1 \
                    if tag not in ner_ns_dict[label] \
                    else ner_ns_dict[label][tag] + 1

        for k in ['0', '1']:
            pos_dict[k] = {
                k: v for k, v in sorted(
                    pos_dict[k].items(),
                    key=lambda item: item[1]
                )
            }
            pos_ns_dict[k] = {
                k: v for k, v in sorted(
                    pos_ns_dict[k].items(),
                    key=lambda item: item[1]
                )
            }
            ner_dict[k] = {
                k: v for k, v in sorted(
                    ner_dict[k].items(),
                    key=lambda item: item[1]
                )
            }
            ner_ns_dict[k] = {
                k: v for k, v in sorted(
                    ner_ns_dict[k].items(),
                    key=lambda item: item[1]
                )
            }

        with open(
            os.path.join(
                PROC_DATA_PATH,
                f"{dataset}_{split_dict['label']}_data.json"
            ),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(data_dict, f)


# def does_claim_mean_checkw():
#     import pandas as pd

#     data_path = INPUT_DATA_PATHS['covid_tweets']['train']['filepath']

#     df = pd.read_csv(data_path, sep='\t', index_col=False)

#     # print(df)
#     print(df[df['claim'] == 1]['check_worthiness'].describe())


def combine_debates():
    import pandas as pd

    data_paths = INPUT_DATA_PATHS['political_debates']

    for dsplit_type, dsplit_dict in data_paths.items():
        files = os.listdir(dsplit_dict['folderpath'])

        if dsplit_type == 'test':
            files = [f for f in files if f[-2:] != 'md']

        header = ['i', 'src', 'content']
        if dsplit_type != 'test':
            header.append('worthy')

        df_combined = pd.DataFrame()
        for f in files:
            fsplit = f.split('_')
            debate_date = fsplit[0]
            if fsplit[-1] == 'combined.tsv':
                continue

            df = pd.read_csv(
                f"{dsplit_dict['folderpath']}/{f}",
                sep='\t',
                index_col=False,
                names=header
            )
            df = df[df['src'] != 'SYSTEM']
            df['id'] = df.apply(
                lambda x: f"{debate_date}{x['i']}",
                result_type='expand',
                axis='columns'
            )
            header_rear = ['i', 'id', 'src', 'content']
            if dsplit_type != 'test':
                header_rear.append('worthy')

            df = df[header_rear]
            # print(df[df['src'] == 'QUESTION']['worthy'].describe())
            df_combined = df_combined.append(
                df,
                ignore_index=True,
            )

        df_combined.to_csv(
            dsplit_dict['filepath'],
            sep='\t',
            index=False,
        )


# combine_debates()
preprocess(dataset='political_debates')
# does_claim_mean_checkw()
