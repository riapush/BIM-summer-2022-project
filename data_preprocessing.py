import pandas as pd
import numpy as np
import glob
import re

## обработка таблицы с маппированием
mapping = pd.read_excel('all_columns (1).xlsx', sheet_name=0)
mapping.drop(mapping[mapping['Следует учитывать?'] != 'Да'].index, inplace=True)


## обработка каждой таблицы по отдельности
files = glob.glob('data/*.xlsx')
for file in files:
    table=[]
    xlsx = pd.ExcelFile(file)
    worksheets = xlsx.sheet_names
    for worksheet in worksheets:
        data = pd.read_excel(xlsx, sheet_name=worksheet)
        table.append(data)
    df = pd.concat(table)
    new_col = []
    names = list(mapping['Column name'])
    codes = list(mapping['Название по коду КСИ'])
    for col in df.columns:
        col_l = re.sub(r'(?<!\d)\.\d+$', '', col.lower())
        if col_l in names:
            new_col.append(codes[names.index(col_l)])
        else:
            df.drop(columns=col, inplace=True)

    df.columns = [re.sub(r'(?<!\d)_\d+$', '', x.lower()) for x in new_col]
    filename = file[5:-5]
    df.to_excel('D:\\Git\\GitHub\\GitHub\\pythonProject\\preprocessed data\\' + filename + '.xlsx', index=False, encoding='utf-8')








