import pandas as pd                         # 데이터 분석 패키지
import numpy as np                          # 계산 패키지
import matplotlib.pyplot as plt             # 데이터 시각화 패키지
import seaborn as sns                       # 데이터 시각화 패키지
import xgboost as xgb                       # XGBoost 패키지
from sklearn.model_selection import KFold   # K-Fold CV
import warnings
warnings.filterwarnings(action='ignore') 
import matplotlib
import sklearn

print('pandas         {}'.format(pd.__version__))
print('numpy          {}'.format(np.__version__))
print('matplotlib     {}'.format(matplotlib.__version__))
print('seaborn        {}'.format(sns.__version__))
print('xgboost        {}'.format(xgb.__version__))
print('sklearn        {}'.format(sklearn.__version__))