import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def nan_check(data):
    total = data.isnull().sum().sort_values(ascending=False) # берем все столбцы, суммируем нулл и  сортируем по убыванию
    percent_1 = data.isnull().sum()/data.isnull().count()*100 # делим кол-во нулловых значений на не-нулловые.
    percent_2 = (np.round(percent_1, 2)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    return missing_data


def init_encoder(df, col_names_list):
    d = {}
    for col_name in col_names_list:
        d[col_name] = df[col_name].unique().tolist()
    return d


def one_hot_encoder(df, var_dict):
    for var, vals in var_dict.items():
        for val in vals:
            df[val] = df[var].apply(lambda x: 1 if val in x else 0)
    return df


df = pd.read_excel('./marketdata.xlsx', sheet_name=0)
df.drop_duplicates()
print(nan_check(df))
data = df[df['nTouchpoints']!=0].reset_index().drop('index', axis=1)
data['recent_touchpoint'] = data['touchpoints'].apply(lambda x: x.split()[-1])
data.loc[data['SocialMedia'] == ' ', 'SocialMedia'] = 'U'
data = data.fillna(value={'creditRating': 'New'})
data['num_creditRating'] = data['creditRating'].replace({'New': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
d = init_encoder(data, ['marital', 'segment', 'SocialMedia', 'creditRating'])
one_hot_encoder(data, d)
data.to_excel('./marketdata2.xlsx')
