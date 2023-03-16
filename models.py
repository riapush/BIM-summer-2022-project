from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

data = pd.read_csv('smth1.csv', sep=';').sort_values(by=['кси код класса'])
counter = Counter(data['кси код класса'])

df = []
indexes = []
start_index = 0
for c in counter:
    if counter[c] < 30:
        start_index += counter[c]
        continue
    df = df + [x for x in range(start_index, start_index + counter[c])]
    for i in range(30):
        ind = random.randint(start_index, start_index + counter[c] - 1)
        while ind in indexes:
            ind = random.randint(start_index, start_index + counter[c] - 1)
        indexes.append(ind)
    start_index += counter[c]

balanced_data = data.iloc[indexes]
balanced_data = balanced_data.drop_duplicates()
new_d = data.iloc[df]
new_d = new_d.drop_duplicates()
label = balanced_data['кси код класса']
label.value_counts().plot(kind = 'bar')
plt.savefig('Распределение числа меток (после обработки всего что можно)')
balanced_data = balanced_data.drop(columns='кси код класса')

all_label = new_d['кси код класса']
new_d.drop(columns='кси код класса', inplace=True)


for i in range(1):
    ## Разделение выборки на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(balanced_data, label, test_size = 0.25, stratify=label)

    ## Обучение Наивного Байеса
    #model = MultinomialNB()
    #model.fit(X_train, y_train)

    ## Оценка результатов классификации
    #print('Результат обучения NB:     ' + str(model.score(X_train, y_train)))
    #print('Результат тестирования NB: ' + str(model.score(X_test, y_test)))

    ## Обучение
    model = RandomForestClassifier(n_estimators=300, min_samples_split=3, min_samples_leaf=2)
    model.fit(X_train, y_train)

    ## Оценка результатов классификации
    print('Результат обучения RF:     ' + str(model.score(balanced_data, label)))
    print('Результат тестирования RF: ' + str(model.score(new_d, all_label)))
    print('F1-score RF: ' + str(f1_score(model.predict(new_d), all_label, average='macro')))

    # catboost обучение
    model = CatBoostClassifier(iterations=1000,
                           learning_rate=1,
                           depth=5,
                           loss_function='MultiClass')
    model.fit(X_train, y_train)

    ## Оценка результатов классификации
    print('Результат обучения CatBoost:     ' + str(model.score(balanced_data, label)))
    print('Результат тестирования CatBoost: ' + str(model.score(new_d, all_label)))
    print('CatBoost F1-Score (Macro): ' + str(f1_score(model.predict(new_d), all_label, average='macro')))

    # lightgbm обучение
    model = LGBMClassifier()
    model.fit(X_train, y_train)

    ## Оценка результатов классификации
    print('Результат обучения LightGBM:     ' + str(model.score(balanced_data, label)))
    print('Результат тестирования LightGBM: ' + str(model.score(new_d, all_label)))
    print('LightGBM F1-Score (Macro): ' + str(f1_score(model.predict(new_d), all_label, average='macro')))

    ## Обучение
    model = XGBClassifier(num_class = 'auto')
    model.fit(X_train, y_train)

    ## Оценка результатов классификации
    print('Результат обучения XGB:     ' + str(model.score(balanced_data, label)))
    print('Результат тестирования XGB: ' + str(model.score(new_d, all_label)))
    print('XGB F1-Score (Macro): ' + str(f1_score(model.predict(new_d), all_label, average='macro')))
    print('\n')
