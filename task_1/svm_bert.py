# Features: Word Embeddings
# Models: SVM

import time
import pickle
import json
import argparse
from scipy.sparse import data
from thundersvm import *
from sklearn import svm
import gensim.downloader as api
import gensim
import spacy
import numpy as np
import pandas as pd

from typing import Union
from tqdm import tqdm
from nltk.corpus import stopwords
from bert_embeddings import load_bert_embeddings
from scorer.main import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, ensemble, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys
import os
from definitions import PROC_DATA_PATH, BERT_EMB_PATH, RESULTS_PATH, \
    INPUT_DATA_PATHS

sys.path.append('.')


my_loc = os.path.dirname(__file__)
data_path = os.path.join(my_loc, 'data')


parser = argparse.ArgumentParser(description='Training for Word Embs')
parser.add_argument('--normalize', type=int, default=1,
                    help='0,1')
parser.add_argument('--bert_type', type=int, default=0,
                    help='0,1,2,3')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='0,1,2,3')

args = parser.parse_args()


def get_best_svm_model(
    feature_vector_train,
    label,
    feature_vector_valid,
    fname,
    emb_type,
    val_y,
    valDF,
    dataset_name
):
    # ft_train, train_y,
    #             ft_val, fname,
    #             emb_type, val_y, valDF,
    # param_grid = [{'kernel':'linear', 'C': np.logspace(-2, 2, 10), 'gamma': [1]},
    #               {'kernel':'rbf', 'C': np.logspace(-2, 2, 10),
    #               'gamma': np.logspace(-2, 2, 10)}]
    param_grid = [{'kernel': 'rbf', 'C': np.logspace(-3, 3, 30),
                  'gamma': np.logspace(-3, 3, 30)}]

    pca_list = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]
    best_acc = 0.0
    best_model = 0
    best_prec = 0.0
    best_pca = 0
    temp_xtrain = feature_vector_train
    temp_xval = feature_vector_valid

    for pca_nk in tqdm(pca_list, desc='pca_list'):
        # print(pca_nk)
        if pca_nk != 1.0:
            pca = decomposition.PCA(n_components=pca_nk).fit(temp_xtrain)
            feature_vector_train = pca.transform(temp_xtrain)
            feature_vector_valid = pca.transform(temp_xval)
        for params in tqdm(param_grid, desc='param_grid', leave=False):
            for C in tqdm(params['C'], desc='params[C]', leave=False):
                for gamma in tqdm(
                    params['gamma'],
                    desc='params[gamma]',
                    leave=False
                ):
                    # Model with different parameters
                    # model = SVC(
                    #     C=C,
                    #     gamma=gamma,
                    #     kernel=params['kernel'],
                    #     random_state=42,
                    #     class_weight='balanced',
                    #     # gpu_id=args.gpu_id
                    # )
                    model = svm.SVC(
                        C=C,
                        gamma=gamma,
                        kernel=params['kernel'],
                        random_state=42,
                        class_weight='balanced',
                    )
                    # print('\npost new model\n')
                    # fit the training dataset on the classifier
                    # pf = time.time()
                    model.fit(feature_vector_train, label)
                    # print('\npost fit time (s): ',
                    #   (time.time() - pf)/60.0, '\n')

                    # predict the acc on validation dataset
                    acc = model.score(feature_vector_valid, val_y)
                    # print('\npost score\n')

                    predicted_distance = model.decision_function(
                        feature_vector_valid
                    )
                    # print('\npredicted_dist: ', predicted_distance, '\n')
                    # print('\npost decision fun\n')
                    results_fpath = my_loc + \
                        '/results/bert_word_%s_%s_svm_norm%d.tsv' % (
                            fname, emb_type, args.normalize)

                    with open(results_fpath, "w") as results_file:
                        for i, line in valDF.iterrows():
                            dist = predicted_distance[i]
                            results_file.write(
                                "{}\t{}\t{}\t{}\n".format(
                                    dataset_name,
                                    line['id'],
                                    dist,
                                    "bert_wd"
                                )
                            )
                    # print('\npost write\n')
                    if dataset_name == 'covid_tweets':
                        _, _, avg_precision, _, _ = evaluate(
                            f'{data_path}/dev.tsv', results_fpath)
                    else:
                        _, _, avg_precision, _, _ = evaluate(
                            f"{INPUT_DATA_PATHS[dataset_name]['folderpath']}/val_combined.tsv",
                            results_fpath
                        )
                    # print('\npost eval\n')
                    if round(avg_precision, 4) >= round(best_prec, 4) and round(acc, 2) >= round(best_acc, 2):
                        best_prec = avg_precision
                        best_acc = acc
                        best_model = model
                        best_pca = pca_nk

    return best_acc, best_pca, best_model


def get_tweet_data(tweet_list):
    twit_y, twit_id = [], []
    for id in tweet_list:
        twit_id.append(id)
        twit_y.append(tweet_list[id]['label'])

    tweetDF = pd.DataFrame()
    tweetDF['label'] = twit_y
    tweetDF['id'] = twit_id

    return np.array(twit_y).astype(np.int32), tweetDF


def get_pos_feat(data_dict, pos_type, index=None):
    pos_tags = {'NOUN': 0, 'VERB': 1, 'PROPN': 2, 'ADJ': 3, 'ADV': 4, 'NUM': 5}

    pos_feat = []
    for id in data_dict:
        # TODO: test this

        if index and id not in index:
            continue

        temp = np.zeros(len(pos_tags))
        proc_twit = data_dict[id][pos_type]
        for wd in proc_twit:
            pos = wd.split('_')[1]
            if pos in pos_tags:
                temp[pos_tags[pos]] += 1

        if sum(temp) > 0:
            temp = temp/sum(temp)

        pos_feat.append(temp)

    return np.array(pos_feat)


def get_dep_map(nlp, train_data, dep, index=None):
    pos_tags = {'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'NUM'}
    deps = set()
    for id in train_data:
        if index and id not in index:
            continue

        words = train_data[id][dep]
        sent = " ".join(words)

        doc = nlp(sent)

        for token in doc:
            if token.pos_ in pos_tags and token.head.pos_ in pos_tags:
                # rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_
                rel = token.pos_+'-'+token.dep_
                deps.add(rel)

    return {val: i for i, val in enumerate(list(deps))}


def get_dep_feats(nlp, dep_map, data_dict, dep, index=None):
    feats = []
    for id in data_dict:
        if index and id not in index:
            continue

        temp = np.zeros(len(dep_map))
        words = data_dict[id][dep]
        sent = " ".join(words)

        doc = nlp(sent)

        for token in doc:
            # rel = token.pos_+'-'+token.dep_+'-'+token.head.pos_
            rel = token.pos_+'-'+token.dep_
            if rel in dep_map:
                temp[dep_map[rel]] += 1

        if sum(temp) > 0:
            temp = temp/sum(temp)

        feats.append(temp)

    return np.array(feats)


def svm_bert(
    dataset: str = 'covid_tweets',
    pos: Union[str, bool] = False,
    dep: Union[str, bool] = False,
    ensemble: bool = False,
):

    # train_y, trainDF = get_tweet_data(train_dict)
    # val_y, valDF = get_tweet_data(val_dict)

    all_res = []

    # files = ['raw_bert-large-uncased']
    # fname = 'bert-large-uncased_raw_text'

    # data = json.load(open(f'{BERT_EMB_PATH}/{dataset}_{fname}.json', 'r'))

    emb_list = [
        'sent_word_catavg_wostop',
        'sent_word_sumavg_wostop',
        'sent_emb_2_last_wostop',
        'labels'
    ]
    txt_type, bert_type = 'raw', 'bert-large-uncased'

    embeddings = {
        k: prepare(v, dataset='political_debates', subset=0.025)
        for k, v in load_bert_embeddings(
            dataset,
            text_type=txt_type,
            bert_type=bert_type,
            emb_cat=emb_list
        ).items()
    }

    train_dict, tindex, vindex = [None]*3
    if pos:
        train_dict = json.load(
            open(f'{PROC_DATA_PATH}/{dataset}_train_data.json', 'r')
        )
        if dataset == 'covid_tweets':
            val_dict = json.load(
                open(f'{PROC_DATA_PATH}/{dataset}_val_data.json', 'r')
            )
            val_pos = get_pos_feat(val_dict, pos)
            train_pos = get_pos_feat(train_dict, pos)

        else:
            tindex = set(
                embeddings['labels']['train']['id'].values.astype(str)
            )
            vindex = set(
                embeddings['labels']['val']['id'].values.astype(str)
            )

            train_pos = get_pos_feat(train_dict, pos, tindex)
            val_pos = get_pos_feat(train_dict, pos, vindex)

    if dep:
        if not train_dict:
            train_dict = json.load(
                open(f'{PROC_DATA_PATH}/{dataset}_train_data.json', 'r')
            )

        nlp = spacy.load('en_core_web_lg')
        dep_map = get_dep_map(nlp, train_dict, dep)

        if dataset == 'covid_tweets':
            val_dict = json.load(
                open(f'{PROC_DATA_PATH}/{dataset}_val_data.json', 'r')
            )
            val_dep = get_dep_feats(nlp, dep_map, val_dict, pos)
            train_dep = get_dep_feats(nlp, dep_map, train_dict, pos)

        else:
            tindex = tindex if tindex else set(
                embeddings['labels']['train']['id'].values.astype(str)
            )
            vindex = vindex if vindex else set(
                embeddings['labels']['val']['id'].values.astype(str)
            )

            train_dep = get_dep_feats(nlp, dep_map, train_dict, dep, tindex)
            val_dep = get_dep_feats(nlp, dep_map, train_dict, dep, vindex)

    # exit()
    train_y = embeddings['labels']['train']['p0'].values
    val_y = embeddings['labels']['val']['p0'].values

    # emb_list = ['sent_word_catavg', 'sent_word_catavg_wostop', 'sent_word_sumavg',
    #             'sent_word_sumavg_wostop', 'sent_emb_2_last', 'sent_emb_2_last_wostop',
    #             'sent_emb_last', 'sent_emb_last_wostop']
    if ensemble:
        pred_all, desc_all = [], []

    fname = f'{txt_type}_{bert_type}'
    for emb_type in emb_list:
        since = time.time()

        if emb_type == 'labels':
            continue

        split_emb = embeddings[emb_type]
        # emb_type_df = embeddings[emb_type]
        # train_emb_type_df = emb_type_df[emb_type_df['split_type'] == 'train']

        # if dataset == 'political_debates':
        #     val_emb_type_df = train_emb_type_df[mask].reset_index(drop=True)
        #     train_emb_type_df = train_emb_type_df[~mask].reset_index(drop=True)
        # else:
        #     val_emb_type_df = emb_type_df[emb_type_df['split_type'] == 'val']

        ft_train = split_emb['train'].iloc[:, 2:]
        ft_val = split_emb['val'].iloc[:, 2:]

        valDF = pd.DataFrame(val_y, columns=['label'])
        valDF['id'] = split_emb['val']['id']
        valDF = valDF.astype(int)

        if args.normalize:
            tr_norm = np.linalg.norm(ft_train, axis=1)
            tr_norm[tr_norm == 0] = 1.0

            val_norm = np.linalg.norm(ft_val, axis=1)
            val_norm[val_norm == 0] = 1.0

            ft_train = ft_train/tr_norm[:, np.newaxis]
            ft_val = ft_val/val_norm[:, np.newaxis]

        if pos:
            ft_train = np.concatenate((ft_train, train_pos), axis=1)
            ft_val = np.concatenate((ft_val, val_pos), axis=1)

        if dep:
            ft_train = np.concatenate((ft_train, train_dep), axis=1)
            ft_val = np.concatenate((ft_val, val_dep), axis=1)

        if ensemble:
            model_path_no_ext = os.path.join(
                my_loc,
                'models',
                f'{fname}_{emb_type}_norm{args.normalize}'
            )

            with open(f'{model_path_no_ext}.pkl', 'rb') as mb:
                model_params = pickle.load(mb)
                best_pca = model_params['best_pca']

            classifier = svm.SVC()
            with open(
                f'{model_path_no_ext}.dt',
                'rb'
            ) as cb:
                classifier = pickle.load(cb)

        else:
            accuracy, best_pca, classifier = get_best_svm_model(
                ft_train, train_y,
                ft_val, fname,
                emb_type, val_y, valDF,
                dataset_name=dataset
            )

        if best_pca != 1.0:
            pca = decomposition.PCA(n_components=best_pca).fit(ft_train)
            ft_val = pca.transform(ft_val)

        if ensemble:
            print(
                f'Model {emb_type} ACC: {classifier.score(ft_val, val_y):.3f}'
            )

            pred_all.append(classifier.predict(ft_val))
            desc_all.append(classifier.decision_function(ft_val))

        else:
            print("SVM, %s, %s Accuracy: %.3f" %
                  (fname, emb_type, round(accuracy, 3)))
            print("PCA No. Components: %.2f, Dim: %d" %
                  (best_pca, ft_val.shape[1]))
            print("C: %.3f, Gamma: %.3f, kernel: %s" %
                  (classifier.C, classifier.gamma, classifier.kernel))

            predicted_distance = classifier.decision_function(ft_val)
            results_fpath = f'{RESULTS_PATH}/{dataset}_bert_word_{fname}_{emb_type}_svm_norm{args.normalize}.tsv'
            with open(results_fpath, "w") as results_file:
                for i, line in valDF.iterrows():
                    dist = predicted_distance[i]
                    results_file.write("{}\t{}\t{}\t{}\n".format(
                        dataset,
                        line['id'],
                        dist,
                        'bert_wd'
                    ))

            if dataset == 'covid_tweets':
                _, _, avg_precision, _, _ = evaluate(
                    f'{data_path}/dev.tsv', results_fpath)
            else:
                _, _, avg_precision, _, _ = evaluate(
                    f"{INPUT_DATA_PATHS[dataset]['folderpath']}/val_combined.tsv",
                    results_fpath
                )
            # _, _, avg_precision, _, _ = evaluate(
            #     f'{data_path}/dev.tsv', results_fpath)
            print(
                f"{dataset}, {fname}, {emb_type} SVM AVGP: {round(avg_precision, 4)}\n")
            # print('best_pca', best_pca)
            with open(
                my_loc+'/models/'+fname+'_'+emb_type +
                    '_norm%s.pkl' % (args.normalize),
                'wb'
            ) as bpcaf:
                pickle.dump({'best_pca': best_pca}, bpcaf)

            with open(
                my_loc+'/models/'+fname + '_'+emb_type +
                    '_norm%s.dt' % (args.normalize),
                'wb'
            ) as bmodelf:
                pickle.dump(classifier, bmodelf)

            all_res.append([emb_type, round(accuracy, 3), round(avg_precision, 4),
                            best_pca, ft_train.shape[1], ft_val.shape[1]])

            print("Completed in: {} minutes\n".format((time.time()-since)/60.0))

    if ensemble:
        final_pred = np.ceil(np.mean(pred_all, axis=0)).astype(np.int8)
        final_desc = np.zeros(len(final_pred))
        final_desc[final_pred == 1] = np.max(desc_all, axis=0)[final_pred == 1]
        final_desc[final_pred == 0] = np.min(desc_all, axis=0)[final_pred == 0]

        print("Ensemble ACC: %.3f" % (sum(val_y == final_pred)/len(val_y)))

        # results_fpath = my_loc+'/results/task1_ensemble_bert_posdep_svm_dev.tsv'
        results_fpath = os.path.join(
            my_loc,
            'results',
            'task1_ensemble_bert_posdep_svm_dev.tsv'
        )
        with open(results_fpath, "w") as results_file:
            for i, line in valDF.iterrows():
                dist = final_desc[i]
                results_file.write(
                    "{}\t{}\t{}\t{}\n".format(dataset, line['id'],
                                              dist, "bert_posdep"))

        if dataset == 'covid_tweets':
            _, _, avg_precision, _, _ = evaluate(
                f'{data_path}/dev.tsv', results_fpath)
        else:
            _, _, avg_precision, _, _ = evaluate(
                f"{INPUT_DATA_PATHS[dataset]['folderpath']}/val_combined.tsv",
                results_fpath
            )

        print("Ensemble Precision: %.4f" % (avg_precision))

    else:
        with open(my_loc+'/file_results/bert_svm_word_%s_norm%d.txt' % (fname, args.normalize), 'w') as f:
            for res in all_res:
                f.write("%s\t%.3f\t%.4f\t%.2f\t%d\t%d\n" %
                        (res[0], res[1], res[2], res[3], res[4], res[5]))

            f.write('\n\n')


def prepare(df, dataset='covid_tweets', subset=1.0, rs=22):

    train = df[df['split_type'] == 'train']
    test = df[df['split_type'] == 'test']

    if dataset == 'covid_tweets':
        val = df[df['split_type'] == 'val']
    else:
        from sklearn.model_selection import train_test_split
        tmask, vmask = train_test_split(
            train['id'].values, test_size=0.25, random_state=rs)

        val = train[train['id'].isin(vmask)].reset_index(drop=True)
        val['split_type'] = 'val'

        train = train[train['id'].isin(tmask)].reset_index(drop=True)

    path = f"{INPUT_DATA_PATHS[dataset]['folderpath']}/val_combined.tsv"
    if dataset == 'political_debates' \
            and df.columns.size < 4 \
            and not os.path.exists(path):

        val_for_eval = val.rename(
            columns={'p0': 'label'}
        ).astype(
            {'label': int}
        )
        # val_for_eval
        val_for_eval['src'] = '?'
        val_for_eval['content'] = '?'
        val_for_eval = val_for_eval[['id', 'src', 'content', 'label']]
        val_for_eval.to_csv(
            path,
            sep='\t'
        )

    if subset < 1.0:
        train = train.sample(frac=subset, random_state=rs).reset_index(
            drop=True
        )
        val = val.sample(frac=subset, random_state=rs).reset_index(drop=True)
        test = test.sample(frac=subset*0.8, random_state=rs).reset_index(
            drop=True
        )

    return {'train': train, 'val': val, 'test': test}


# def subset(df):
pos_type = 'pos_ns'  # equivalent to pos_twit_nostop
dep_type = 'cleaned_ns'
svm_bert(dataset='political_debates',
         pos=pos_type, dep=dep_type, ensemble=True)

# labels = embeddings['labels']
# for i in emb_list[:-1]:
#     typ = embeddings[i]

#     print(f'train comparison labels vs {i}: ')
#     print(labels['train']['id'].compare(typ['train']['id']))
#     print(labels['train'])
#     print(typ['train'])
#     print(f'val comparison labels vs {i}: ')
#     print(labels['val']['id'].compare(typ['val']['id']))
#     print(labels['val'])
#     print(typ['val'])

# print(embeddings[])

# labels = embeddings['labels'].dropna()
# trainDF = labels[labels['split_type'] == 'train'][['p0', 'id']].rename(
#     columns={'p0': 'label'}
# ).astype(
#     {'label': int}
# )
# mask = None
# if dataset == 'political_debates':  # political_debates
#     from sklearn.model_selection import train_test_split

#     _, val_ids = train_test_split(
#         trainDF['id'].values,
#         test_size=0.2,
#         random_state=2
#     )
#     mask = trainDF['id'].isin(val_ids)
#     valDF = trainDF[mask].reset_index(drop=True)
#     trainDF = trainDF[~mask].reset_index(drop=True)

# else:
#     valDF = labels[labels['split_type'] == 'val'][['p0', 'id']].rename(
#         columns={'p0': 'label'}
#     ).astype(
#         {'label': int}
#     )
