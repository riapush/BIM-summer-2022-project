from collections import Counter
import pandas as pd
import re
import glob
import numpy as np

## Приведение строки к нижнему регистру удаление пробелов и концевых символов  вида ".1"-".9"
def transform(s):
    return re.sub(r'(?<!\d)\.\d+$', '', s.lower())

## Чтение файлов
files = glob.glob('preprocessed data/*.xlsx')
worksheets_dfs = []
for file in files:
    xlsx = pd.ExcelFile(file)
    worksheets = xlsx.sheet_names
    for worksheet in worksheets:
        data = pd.read_excel(xlsx, sheet_name = worksheet)
        ## Приведение заголовков к нижнему регистру и удаление номеров для повторяющихся
        data.columns = [transform(x) for x in data.columns]

        ## Подсчет повторений заголовка
        counter = Counter(data.columns)

        ## Обработка столбцов с дублирующимися заголовками
        doubles = {element: count for element, count in counter.items() if count > 1}
        for d in doubles:
            tmp = data[d].iloc[:, 0]
            for c in range(1, doubles[d]):
                for i in range(0, len(tmp)):
                    ## Объединение информации из столбцов по принципу заполнения пустых ячеек и игнорирования уже заполненных
                    ## Особый случай - столбец description, он должен cодержать код
                    if (pd.isnull(tmp[i]) and not pd.isnull(data[d].iloc[:, c][i])) \
                            or (d == 'КСИ Код класса'.lower() and (type(tmp[i]) != str or len(tmp[i]) < 3 or len(tmp[i]) > 11)):
                        tmp[i] = data[d].iloc[:, c][i]
            ## Замена всех столбцов с общим заголовком одним полученным столбцом
            data.drop(columns=d, inplace=True)
            data[d] = tmp
        worksheets_dfs.append(data)

## Объединение всех листов в один
worksheets_data = pd.concat(worksheets_dfs)

## Удаление столбцов заполненных менее чем на 75%
# columns = worksheets_data.columns
# to_drop = []
# for c in columns:
#     if pd.isnull(worksheets_data[c]).sum() / worksheets_data.shape[0] > 0.25:
#         to_drop.append(c)
# worksheets_data.drop(columns = to_drop, inplace=True)

## Удаление столбцов tag и type id
# worksheets_data.drop(columns = ['tag', 'container name', 'name', 'tag (type)', 'object type'], inplace = True)

## Удаление пробелов, точек, нижних подчеркиваний и дефисов и приведение к нижнему регистру текста в ячейках
## Числа в составе текста не изменяются
# for i in range(worksheets_data.shape[0]):
#     for j in range(worksheets_data.shape[1]):
#         if type(worksheets_data.iat[i, j]) == str:
#             worksheets_data.iat[i, j] = re.sub(r'((?<!\d)\.)|((-|\.)(?!\d))|((?<=[a-zA-Zа-яА-Я)])-)|_|:.*$', '',
#                 worksheets_data.iat[i, j].lower()).replace(' ','').replace("''",'"').replace('notdefined', '')
#             worksheets_data.iat[i, j] = re.sub(r'(?<=\d)х(?=\d)', 'x', worksheets_data.iat[i, j])

## Удаление столбцов заполненных менее чем на 75%, которые остались после удаления notdefined
# columns = worksheets_data.columns
# to_drop = []
# for c in columns:
#     if pd.isnull(worksheets_data[c]).sum() / worksheets_data.shape[0] > 0.25:
#         to_drop.append(c)
# worksheets_data.drop(columns=to_drop, inplace=True)

# worksheets_data['КСИ Код класса'].replace('', np.nan, inplace=True)
# worksheets_data.dropna(subset=['КСИ Код класса'], inplace=True)
# worksheets_data['КСИ Код класса'].apply(lambda x: x.replace(x, x[0:3]) if x[-1].isdigit() else x)

## Удаление дублирующихся столбцов и строк
worksheets_data = worksheets_data.T.drop_duplicates().T.drop_duplicates()

## Запись результата в файл
worksheets_data.to_csv('general.csv', index=False, encoding='utf-8')