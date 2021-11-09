# from sklearn.externals import joblib
import os
import pandas as pd
import nltk
import pickle

from sklearn import svm
from sklearn import datasets

# nltk.download('stopwords')

# df = pd.DataFrame([{'test': 10, 'test2': 20}])

# print(df)

# model = svm.SVC()

# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# model.fit(X, y)
# # print(model.get_params())

# # # joblib.dump(model, 'model-joblib.pkl')

# with open(
#     f'{os.getcwd()}/model-pickle.dt',
#     'wb'
# ) as bmodelf:
#     pickle.dump(model, bmodelf)

# # loaded_joblib = joblib.load(f'{os.getcwd()}/model-joblib.pkl')
# # print(loaded_joblib)

# with open(
#     f'{os.getcwd()}/model-pickle.dt',
#     'rb'
# ) as mpf:
#     loaded_pickle = pickle.load(mpf)

#     print(loaded_pickle.get_params())
#     print(loaded_pickle.predict(X)[0])
#     print(y[0])

# from definitions import BERT_EMB_PATH

# with open(
#     f'{BERT_EMB_PATH}/political_debates_bert-base-uncased_proc_text.json',
#     'r'
# ) as bembf:
#     print(bembf.readline())
