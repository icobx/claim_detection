import os
import pandas as pd

from definitions import EMB_DF_PATHS

txt_types = ['raw', 'processed']
bert_types = ['bert-large-uncased', 'bert-base-uncased']

for txt_type in txt_types:
    for bert_type in bert_types:
        fn = f'{txt_type}_{bert_type}_labels.csv'
        print(fn)

        fp = os.path.join(EMB_DF_PATHS['covid_tweets'], fn)
        df = pd.read_csv(
            fp,
            names=['id', 'phase', 'label']
        )

        labels_next_to_test = df[df['phase'] == 'test']['label']
        labels_next_to_val = df[df['phase'] == 'val']['label'].dropna()

        val_labels = labels_next_to_test.append(labels_next_to_val)

        try:
            df.loc[df['phase'] == 'val', 'label'] = val_labels.values
        except:
            continue
        df.loc[df['phase'] == 'test', 'label'] = None

        df.to_csv(fp, header=False, index=False)
