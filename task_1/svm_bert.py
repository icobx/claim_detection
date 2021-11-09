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

from tqdm import tqdm
from nltk.corpus import stopwords
from bert_embeddings import load_bert_embeddings
from scorer.main import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, ensemble, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
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
    # param_grid = [{'kernel':'linear', 'C': np.logspace(-2, 2, 10), 'gamma': [1]},
    #               {'kernel':'rbf', 'C': np.logspace(-2, 2, 10),
    #               'gamma': np.logspace(-2, 2, 10)}]
    param_grid = [{'kernel': 'rbf', 'C': np.logspace(-3, 3, 30),
                  'gamma': np.logspace(-3, 3, 30)}]

    pca_list = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]
    best_acc = 0.0
    best_model = 0
    best_prec = 0.0
    best_pca_nk = 0
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
                    print('post new model')
                    # fit the training dataset on the classifier
                    pf = time.time()
                    model.fit(feature_vector_train, label)
                    print('post fit time (s): ', (time.time() - pf)/60.0)

                    # predict the acc on validation dataset
                    acc = model.score(feature_vector_valid, val_y)
                    print('post score')

                    predicted_distance = model.decision_function(
                        feature_vector_valid
                    )
                    print('post decision fun')
                    results_fpath = my_loc + \
                        '/results/bert_word_%s_%s_svm_norm%d.tsv' % (
                            fname, emb_type, args.normalize)
                    # TODO: TEMPORARY SOLUTION
                    temp = valDF
                    temp['src'] = 'src'
                    temp['content'] = 'content'
                    temp = temp[['id', 'src', 'content', 'label']]
                    temp.to_csv(
                        f"{INPUT_DATA_PATHS[dataset_name]['folderpath']}/temp_val_combined.tsv",
                        sep='\t'
                    )
                    # i, id, src, content, label
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
                    print('post write')
                    if dataset_name == 'covid_tweets':
                        _, _, avg_precision, _, _ = evaluate(
                            f'{data_path}/dev.tsv', results_fpath)
                    else:
                        _, _, avg_precision, _, _ = evaluate(
                            f"{INPUT_DATA_PATHS[dataset_name]['folderpath']}/temp_val_combined.tsv",
                            results_fpath
                        )
                    print('post eval')
                    if round(avg_precision, 4) >= round(best_prec, 4) and round(acc, 2) >= round(best_acc, 2):
                        best_prec = avg_precision
                        best_acc = acc
                        best_model = model
                        best_pca_nk = pca_nk

    return best_acc, best_pca_nk, best_model


def get_tweet_data(tweet_list):
    twit_y, twit_id = [], []
    for id in tweet_list:
        twit_id.append(id)
        twit_y.append(tweet_list[id]['label'])

    tweetDF = pd.DataFrame()
    tweetDF['label'] = twit_y
    tweetDF['id'] = twit_id

    return np.array(twit_y).astype(np.int32), tweetDF


def svm_bert(dataset: str = 'covid_tweets'):

    # train_dict = json.load(
    #     open(f'{PROC_DATA_PATH}/{dataset}_train_data.json', 'r')
    # )
    # val_dict = json.load(
    #     open(f'{PROC_DATA_PATH}/{dataset}_val_data.json', 'r')
    # )

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
    embeddings = load_bert_embeddings(
        dataset,
        text_type=txt_type,
        bert_type=bert_type,
        emb_cat=emb_list
    )
    labels = embeddings['labels'].dropna()
    trainDF = labels[labels['split_type'] == 'train'][['p0', 'id']].rename(
        columns={'p0': 'label'}
    ).astype(
        {'label': int}
    )
    mask = None
    if dataset == 'political_debates':  # political_debates
        from sklearn.model_selection import train_test_split

        _, val_ids = train_test_split(
            trainDF['id'].values,
            test_size=0.2,
            random_state=2
        )
        mask = trainDF['id'].isin(val_ids)
        valDF = trainDF[mask].reset_index(drop=True)
        trainDF = trainDF[~mask].reset_index(drop=True)

    else:
        valDF = labels[labels['split_type'] == 'val'][['p0', 'id']].rename(
            columns={'p0': 'label'}
        ).astype(
            {'label': int}
        )

    # exit()
    train_y, val_y = trainDF['label'].values, valDF['label'].values

    # emb_list = ['sent_word_catavg', 'sent_word_catavg_wostop', 'sent_word_sumavg',
    #             'sent_word_sumavg_wostop', 'sent_emb_2_last', 'sent_emb_2_last_wostop',
    #             'sent_emb_last', 'sent_emb_last_wostop']
    fname = f'{txt_type}_{bert_type}'
    for emb_type in emb_list:
        since = time.time()

        emb_type_df = embeddings[emb_type]
        train_emb_type_df = emb_type_df[emb_type_df['split_type'] == 'train']

        if dataset == 'political_debates':
            val_emb_type_df = train_emb_type_df[mask].reset_index(drop=True)
            train_emb_type_df = train_emb_type_df[~mask].reset_index(drop=True)
        else:
            val_emb_type_df = emb_type_df[emb_type_df['split_type'] == 'val']

        ft_train = train_emb_type_df.iloc[:, 2:]
        ft_val = val_emb_type_df.iloc[:, 2:]

        if args.normalize:
            tr_norm = np.linalg.norm(ft_train, axis=1)
            tr_norm[tr_norm == 0] = 1.0

            val_norm = np.linalg.norm(ft_val, axis=1)
            val_norm[val_norm == 0] = 1.0

            ft_train = ft_train/tr_norm[:, np.newaxis]
            ft_val = ft_val/val_norm[:, np.newaxis]

        # train_y =
        accuracy, best_pca_nk, classifier = get_best_svm_model(
            ft_train, train_y,
            ft_val, fname,
            emb_type, val_y, valDF,
            dataset_name=dataset
        )

        if best_pca_nk != 1.0:
            pca = decomposition.PCA(n_components=best_pca_nk).fit(ft_train)
            ft_val = pca.transform(ft_val)

        print("SVM, %s, %s Accuracy: %.3f" %
              (fname, emb_type, round(accuracy, 3)))
        print("PCA No. Components: %.2f, Dim: %d" %
              (best_pca_nk, ft_val.shape[1]))
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

        _, _, avg_precision, _, _ = evaluate(
            f'{data_path}/dev.tsv', results_fpath)
        print(f"{dataset}, {fname}, {emb_type} SVM AVGP: {round(avg_precision, 4)}\n")
        # print('best_pca', best_pca_nk)
        with open(
            my_loc+'/models/'+fname+'_'+emb_type +
                '_norm%s.pkl' % (args.normalize),
            'wb'
        ) as bpcaf:
            pickle.dump({'best_pca': best_pca_nk}, bpcaf)

        with open(
            my_loc+'/models/'+fname + '_'+emb_type +
                '_norm%s.dt' % (args.normalize),
            'wb'
        ) as bmodelf:
            pickle.dump(classifier, bmodelf)

        all_res.append([emb_type, round(accuracy, 3), round(avg_precision, 4),
                        best_pca_nk, ft_train.shape[1], ft_val.shape[1]])

        print("Completed in: {} minutes\n".format((time.time()-since)/60.0))

    with open(my_loc+'/file_results/bert_svm_word_%s_norm%d.txt' % (fname, args.normalize), 'w') as f:
        for res in all_res:
            f.write("%s\t%.3f\t%.4f\t%.2f\t%d\t%d\n" %
                    (res[0], res[1], res[2], res[3], res[4], res[5]))

        f.write('\n\n')


svm_bert(dataset='political_debates')  # dataset='political_debates'
