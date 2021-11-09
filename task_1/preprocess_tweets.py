from preprocess_data import preprocess
import os
import re
from urlextract import URLExtract
import json
from nltk.corpus import stopwords
import spacy
import pickle
from helper_funcs import *
import sys
sys.path.append('.')

my_loc = os.path.dirname(__file__)

nlp = spacy.load('en_core_web_lg')

split_mp = {'training': 'train', 'dev': 'val', 'test': 'test'}

# text processor specifically for tweets,
# https://github.com/cbaziotis/ekphrasis
text_processor_twit = get_text_processor(word_stats='twitter')

df_stopwords = set(stopwords.words('english'))
# exit()
url_extr = URLExtract()

temp = preprocess()
diff = []

for split in split_mp:
    tr_file = open('%s/data/%s.tsv' % (my_loc, split), 'r')

    data_dict = {}
    cnt = 0
    for line in tr_file:
        if cnt:
            # split line into categories
            if split != 'test':
                topic, id, link, content, claim, worthy = line.strip().split('\t')
            else:
                topic, id, link, content = line.strip().split('\t')
                claim, worthy = 0, 0

            # get urls from tweet text
            urls = url_extr.find_urls(content)

            # break down tweet text and process it,
            # i.e. #CorornaVirus -> <hashtag> corona virus </hashtag>,
            #      @realDonaldTrump -> <user>
            proc_twit = text_processor_twit.pre_process_doc(content)

            # remove strings which are not considered words (i.e. , <user>)
            clean_twit = [
                word for word in proc_twit if not re.search(
                    "[^a-z0-9\s]+",
                    word,
                )
            ]

            clean_twit = [
                word for word in clean_twit
                if len(word) > 2
                or word.isnumeric()
            ]

            clean_twit_nostop = [
                word for word in clean_twit if word not in df_stopwords
            ]

            spacy_twit = nlp(" ".join(clean_twit))
            spacy_twit_nostop = nlp(" ".join(clean_twit_nostop))

            pos_twit, pos_twit_nostop = [], []
            ner_twit, ner_twit_nostop = [], []

            # append coarse (pos_) and fine (tag_) part of speech tags
            for token in spacy_twit:
                pos_twit.append(token.text+'_'+token.pos_+'_'+token.tag_)

            for token in spacy_twit_nostop:
                pos_twit_nostop.append(
                    token.text+'_'+token.pos_+'_'+token.tag_
                )

            # get named entities
            for ent in spacy_twit.ents:
                ner_twit.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })

            for ent in spacy_twit_nostop.ents:
                ner_twit_nostop.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })

            data_dict[id] = {
                'link': link,
                'text': content,
                'twit_proc': proc_twit,
                'twit_clean': clean_twit,
                'twit_clean_nostop': clean_twit_nostop,
                'pos_twit': pos_twit,
                'pos_twit_nostop': pos_twit_nostop,
                'ner_twit': ner_twit,
                'ner_twit_nostop': ner_twit_nostop,
                'claim': claim,
                'worthy': worthy,
                'urls': urls
            }

        cnt += 1

    # Data Stats, PoS and NER
    ner_dict, ner_ns_dict = {'1': {}, '0': {}}, {'1': {}, '0': {}}
    pos_dict, pos_ns_dict = {'1': {}, '0': {}}, {'1': {}, '0': {}}

    for id in data_dict:
        lab = str(data_dict[id]['worthy'])
        for ptag in data_dict[id]['pos_twit']:
            tg = ptag.split('_')[1]
            pos_dict[lab][tg] = 1 if tg not in pos_dict[lab] else pos_dict[lab][tg]+1

        for ptag in data_dict[id]['pos_twit_nostop']:
            tg = ptag.split('_')[1]
            pos_ns_dict[lab][tg] = 1 if tg not in pos_ns_dict[lab] else pos_ns_dict[lab][tg]+1

        for netag in data_dict[id]['ner_twit']:
            tg = netag['label']
            ner_dict[lab][tg] = 1 if tg not in ner_dict[lab] else ner_dict[lab][tg]+1

        for netag in data_dict[id]['ner_twit_nostop']:
            tg = netag['label']
            ner_ns_dict[lab][tg] = 1 if tg not in ner_ns_dict[lab] else ner_ns_dict[lab][tg]+1

    pos_dict['0'] = {k: v for k, v in sorted(
        pos_dict['0'].items(), key=lambda item: item[1])}
    pos_dict['1'] = {k: v for k, v in sorted(
        pos_dict['1'].items(), key=lambda item: item[1])}
    pos_ns_dict['0'] = {k: v for k, v in sorted(
        pos_ns_dict['0'].items(), key=lambda item: item[1])}
    pos_ns_dict['1'] = {k: v for k, v in sorted(
        pos_ns_dict['1'].items(), key=lambda item: item[1])}
    ner_dict['0'] = {k: v for k, v in sorted(
        ner_dict['0'].items(), key=lambda item: item[1])}
    ner_dict['1'] = {k: v for k, v in sorted(
        ner_dict['1'].items(), key=lambda item: item[1])}
    ner_ns_dict['0'] = {k: v for k, v in sorted(
        ner_ns_dict['0'].items(), key=lambda item: item[1])}
    ner_ns_dict['1'] = {k: v for k, v in sorted(
        ner_ns_dict['1'].items(), key=lambda item: item[1])}
    # print(pos_dict)
    # print(pos_ns_dict)
    # print(ner_dict)
    # print(ner_ns_dict)

# for split in temp:
    if split == 'training':
        xx = 'train'
    elif split == 'dev':
        xx = 'val'
    else:
        xx = split

    print(xx)
    dicts = temp[xx]

    pos_dict_pp = dicts['pos_dict']

    print(pos_dict['0'])
    print(temp[xx]['pos_dict']['0'])
    break

    # json.dump(data_dict, open(my_loc+'/data/proc_data/%s_data.json' %
    #           (split_mp[split]), 'w', encoding='utf-8'))
