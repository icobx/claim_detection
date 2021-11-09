import os

from pathlib import Path

PROJECT_PATH = Path(os.path.dirname(__file__))
INPUT_DATA_PATH = os.path.join(PROJECT_PATH.parent.parent.absolute(), 'data')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')

OUTPUT_DATA_PATH = os.path.join(PROJECT_PATH, 'data')
PROC_DATA_PATH = os.path.join(OUTPUT_DATA_PATH, 'proc_data')
BERT_EMB_PATH = os.path.join(OUTPUT_DATA_PATH, 'bert_embs')


INPUT_DATA_PATHS = {
    'covid_tweets': {
        'train': {
            'label': 'train',
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v1',
                'training.tsv',
            ),
        },
        'val': {
            'label': 'val',
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'train',
                'v1',
                'dev.tsv',
            ),
        },
        'test': {
            'label': 'test',
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'covid_tweets',
                'test',
                'test-input.tsv',
            ),
        },
    },
    'political_debates': {
        'folderpath': os.path.join(INPUT_DATA_PATH, 'political_debates'),
        'embpath': os.path.join(BERT_EMB_PATH, 'political_debates'),
        'train': {
            'label': 'train',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'training'
            ),
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'training',
                'train_combined.tsv'
            ),
        },
        'test_no_annotation': {
            'label': 'test_no_annotation',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test_no_annotation'
            ),
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test_no_annotation',
                'test_no_annotation_combined.tsv'
            )
        },
        'test': {
            'label': 'test',
            'folderpath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test'
            ),
            'filepath': os.path.join(
                INPUT_DATA_PATH,
                'political_debates',
                'test',
                'test_combined.tsv'
            )
        }
    }
}

EMB_DF_PATHS = {
    'covid_tweets': os.path.join(BERT_EMB_PATH, 'covid_tweets'),
    'political_debates': os.path.join(BERT_EMB_PATH, 'political_debates'),
}

EMB_CATEGORIES = [
    'sent_word_catavg', 'sent_word_catavg_wostop',
    'sent_word_sumavg', 'sent_word_sumavg_wostop',
    'sent_emb_2_last', 'sent_emb_2_last_wostop',
    'sent_emb_last', 'sent_emb_last_wostop',
    'labels'
]
