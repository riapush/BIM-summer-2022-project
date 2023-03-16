import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from char2vec import Ch2v
import re

## Удаление индексации в конце строки
def transform(s):
    if type(s) == str:
        return re.sub(r':\s*\d+$', '', s)
    else:
        return s

## Чтение файла с объединенными данными
data = pd.read_csv('general.csv')

## Выделение меток
label = data['кси код класса']

## Очистка столбца меток
to_drop = []
i = 0
new_label = []
for l in label:
    if not pd.isna(l):
        l = re.sub(r'(_.*$)|([^a-zA-z0-9])', '', l)
    if l == '' or pd.isna(l):
        to_drop.append(i)
    else:
        new_label.append(l)
    i += 1
data.drop(to_drop, inplace=True)
data['кси код класса'] = new_label

## Чтение описания признаков
df = pd.read_excel('all_columns.xlsx')

## Заполнение пропусков
data = data.fillna(0.0)

## Удаление идентификатора из имени
data['имя'] = [transform(x) for x in data['имя']]

## Удаление дублирующихся строк
data = data.drop_duplicates()

## Приведение типов
bool = [x.lower() for x in df[df['Тип данных'] == 'Булевый']['Название по коду КСИ'].unique()]
for b in bool:
    new_vals = []
    for el in data[b]:
        if type(el) == str:
            if el.lower() == 'false':
                new_vals.append(0)
            else:
                new_vals.append(1)
        else:
            new_vals.append(el)
    data[b] = new_vals

num = [x.lower() for x in df[df['Тип данных'] == 'Число']['Название по коду КСИ'].unique()]
to_drop = []
for n in num:
    for i in data.index:
        if type(data[n][i]) == str and i not in to_drop:
            to_drop.append(i)
data.drop(to_drop, inplace=True)

## Удаление дублирующихся строк
data = data.drop_duplicates()

## Создание копии для сохранения исходного вида
selected = data.copy()

## Кодирование категорий простой нумерацией
labelencoder = LabelEncoder()
to_le = ['предел огнестойкости', 'гост/ту изделия', 'тип по восприятию нагрузки']
data[to_le] = data[to_le].astype(str)
for c in to_le:
    data[c] = labelencoder.fit_transform(data[c])

## Встаривание для текстовых полей
ch2v = Ch2v()
j = 0
to_ch2v = [x.lower() for x in df[df['Тип данных'] == 'Текст']['Название по коду КСИ'].unique()]
to_ch2v.remove('кси код класса')
for c in to_ch2v:
    col = data[c]
    data.drop(columns=c, inplace=True)
    emb = []
    for el in col:
        emb.append(ch2v.encode(el))
    for i in range(20):
        data.insert(i + 20 * j, c + str(i), [x[i] for x in emb])
    j += 1

## Подсчет числа меток каждого типа
new_label = data['кси код класса']
pd.DataFrame(new_label).value_counts().to_excel('num.xlsx')
pd.DataFrame(new_label).value_counts().plot(kind='bar')
plt.savefig('Распределение числа меток (все)')

## Удаление меток из набора данных
data.drop(columns='кси код класса', inplace=True)

## Оценка значимости каждого признака
clf = RandomForestClassifier()
clf.fit(data, new_label)
plt.figure(figsize=(12, 12))
plt.bar(data.columns, clf.feature_importances_)
plt.xticks(rotation=90)
plt.savefig('Значимость признаков (все)')

## Выявление наиболее значимых признаков
features = dict(zip(data.columns, clf.feature_importances_))
importances1 = {}
i = 0
for f in features:
    if i < 20 * len(to_ch2v):
        if re.sub(r'\s*\d+$', '', f) in importances1:
            importances1[re.sub(r'\s*\d+$', '', f)] += features[f]
        else:
            importances1[re.sub(r'\s*\d+$', '', f)] = features[f]
    else:
        importances1[f] = features[f]
importances2 = {}
i = 0
for f in features:
    if i < 20 * len(to_ch2v):
        if re.sub(r'\s*\d+$', '', f) in importances2:
            importances2[re.sub(r'\s*\d+$', '', f)] += features[f] / 20
        else:
            importances2[re.sub(r'\s*\d+$', '', f)] = features[f] / 20
    else:
        importances2[f] = features[f]

## Отбор наиболее значимых признаков
selected_features1 = list(dict(sorted(importances1.items(), key = lambda x: x[1])[-10:]))
selected_features2 = list(dict(sorted(importances2.items(), key = lambda x: x[1])[-10:]))

## Отбор столбцов в закодированном виде
selected_columns1 = []
for c in data.columns:
    if re.sub(r'\s*\d+$', '', c) in selected_features1:
        selected_columns1.append(c)
selected_columns2 = []
for c in data.columns:
    if re.sub(r'\s*\d+$', '', c) in selected_features2:
        selected_columns2.append(c)

## Сохранение отобранных признаков
data1 = data[selected_columns1]
selected_features1.append('кси код класса')
selected1 = selected[selected_features1]
selected1 = selected1.drop_duplicates()
data1['кси код класса'] = new_label
data1 = data1.drop_duplicates()

data2 = data[selected_columns2]
selected_features2.append('кси код класса')
selected2 = selected[selected_features2]
selected2 = selected2.drop_duplicates()
data2['кси код класса'] = new_label
data2 = data2.drop_duplicates()

## Запись преобразованных данных в файл
data1.to_csv('smth1.csv', index=False, sep=';')
data2.to_csv('smth2.csv', index=False, sep=';')

## Запись отобранных данных в исходном виде в файл
selected1.to_excel('selected1.xlsx', index=False)
selected2.to_excel('selected2.xlsx', index=False)