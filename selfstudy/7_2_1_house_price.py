import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

house_df_org = pd.read_csv(r'C:\Users\bitcamp\Desktop\datasets\house-prices-advanced-regression-techniques/house_price.csv')
house_df = house_df_org.copy()
print(house_df.head(3))

print('데이터 세트의 shape:', house_df.shape)
print('\n전체 피처의 type \n', house_df.dtypes.value_counts())
isnull_series = house_df.isnull().sum()
print('\n Null 칼럼과 그 건수 :\n', isnull_series[isnull_series>0].sort_values(ascending=False) )

# plt.title('Original Sale Price Histogram')
# sns.distplot(house_df['SalePrice'])

plt.title('Log Transformed Sale Price Histogram')
log_SalePrice = np.log1p(house_df['SalePrice'])
sns.distplot(log_SalePrice)
plt.show()

# SalePrice 로그 변환
original_SalePrice = house_df['SalePrice']
