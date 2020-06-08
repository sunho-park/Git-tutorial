import pandas as pd                         # 데이터 분석 패키지
import numpy as np                          # 계산 패키지
from sklearn.model_selection import KFold   # K-Fold CV
import warnings
warnings.filterwarnings(action='ignore') 
import sklearn


train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0, encoding='UTF8')
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0, encoding='UTF8')

submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)
