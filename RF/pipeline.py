import pandas as pd
from model_managing import ModelManager
from text_preprocessing import TextPreprocessor
from json_to_df import json_to_df

# Выполнение предсказания
def make_prediction(request, common_info):
    # Создаем менеджера с параметрами по умолчанию
    mm = ModelManager()
    # Предполагается изменение реализации load на зугрузку из БД
    # Пока предполагается, что используется векторизатор по умолчанию, но, если нужно, и его можно подгружать в load
    mm.load(common_info['classifier_id'])
    # Получаем данные из запроса
    json_data = request['body']   
    data_for_prediction = json_to_df(json_data, [('Element_ID', 0), ('XNT_0001', 1), ('XPG_0010', 2), ('XNT_0006', 3),
        ('XNT_0004', 4), ('XNT_0003', 5), ('XNT_0005', 6), ('XPM_0002', 7), ('SN_0003', 8), ('XNT_0002', 9),
        ('XNT_0002', 10)])
    # Убираем то, что не является признаком
    to_vector = list(data_for_prediction.columns)
    if '00_Element_ID' in data_for_prediction.columns:
        to_vector.remove('00_Element_ID')
    # Получаем предсказание
    result = mm.predict(data_for_prediction.loc[:,to_vector])
    # Приводим к нужному виду
    result['Element_ID'] = data_for_prediction['00_Element_ID']
    return result.set_index('Element_ID').T.to_dict()


# Расширение датасета
def extend_train_data(json_data, common_info):
    data_for_adding = json_to_df(json_data, [('Element_ID', 0), ('XNT_0001', 1), ('XPG_0010', 2), ('XNT_0006', 3),
        ('XNT_0004', 4), ('XNT_0003', 5), ('XNT_0005', 6), ('XPM_0002', 7), ('SN_0003', 8), ('XNT_0002', 9),
        ('XNT_0002', 10), ('XNKC0001', 11)]) # Последнее - кси код класса

    '''Тут часть текущей реализации, смысл которой мне не понятен.
    '''

    # Инициализация векторизатора по умолчанию
    tp = TextPreprocessor()
    tp.init_ch2v()
    # Векторизация полученных данных
    to_vector = list(data_for_adding.columns)
    to_vector.remove('00_Element_ID')
    to_vector.remove('11_XNKC0001')
    
    '''Эта часть зависит от БД, поэтому только идея.
    Тут нужно сохранение векторизации по типу того, что реализовано внутри _lemmatise_text.
    Получить векторизацию можно следующим образом vectorisation = tp.encode_df(data.loc[:,to_vector]),
    где data - некоторое подмножество строк датафрейма, для которого еще нет векторизации.
    Но работать уже нужно не с одним столбцом, а с 10. Для "очистки" ячеек нужно использовать tp.clear_text.
    '''

    '''Тут сохранение доп информации и вызов stats = mgdbo.update_train_data(...).
    '''
    stats = None

    _update_model(common_info['classifier_id'])        
    return stats


# Простое обновление модели
def _update_model(classifier_id):

    '''Получить датасет из БД -> data, classes
    '''
    data = [] #<- векторизованные данные
    classes = []

    mm = ModelManager()
    # Предполагается изменение реализации load на зугрузку из БД
    # Пока предполагается, что используется векторизатор по умолчанию, но, если нужно, и его можно подгружать в load
    mm.load(classifier_id)

    old_model = mm.train(data, classes)

    '''Получить данные для тестирования и выбора лучшей модели -> test_data, test_classes
    '''
    test_data = [] #<- векторизованные данные
    test_classes = []

    mm.choose_best(old_model, test_data, test_classes)

    # Предполагается изменение реализации save на запись в БД
    # Пока предполагается, что векторизатор не сохраняется, но, если нужно, и его можно сохранять в save
    mm.save()


# Обучение с калибровкой (по идее требуется один раз для новой/некалиброванной модели, дальше будет уже загружаться
# и обучаться модель с доп надстройкой, которая и отвечает за калибровку)
def _calibrate(classifier_id):
    
    '''Получить датасет из БД -> data, classes.
    '''
    data = [] #<- векторизованные данные
    classes = []

    mm = ModelManager()
    # Предполагается изменение реализации load на зугрузку из БД
    # Пока предполагается, что используется векторизатор по умолчанию, но, если нужно, и его можно подгружать в load
    mm.load(classifier_id)

    old_model = mm.calibrate(data, classes, cv=3)

    '''Получить данные для тестирования и выбора лучшей модели -> test_data, test_classes.
    '''
    test_data = [] #<- векторизованные данные
    test_classes = []

    mm.choose_best(old_model, test_data, test_classes)

    # Предполагается изменение реализации save на запись в БД
    # Пока предполагается, что векторизатор не сохраняется, но, если нужно, и его можно сохранять в save
    mm.save()


# Обучение с подбором гиперпараметров (по идее требуется редко - для новой модели или если метрики заметно ухудшились)
def _optimize_hyper(classifier_id):
    
    '''Получить датасет из БД -> data, classes
    '''
    data = [] #<- векторизованные данные
    classes = []

    mm = ModelManager()
    # Предполагается изменение реализации load на зугрузку из БД
    # Пока предполагается, что используется векторизатор по умолчанию, но, если нужно, и его можно подгружать в load
    mm.load(classifier_id)

    old_model = mm.optimize_hyper(data, classes, grid=None, n_iter=50, cv=3)

    '''Получить данные для тестирования и выбора лучшей модели -> test_data, test_classes
    '''
    test_data = [] #<- векторизованные данные
    test_classes = []

    mm.choose_best(old_model, test_data, test_classes)

    # Предполагается изменение реализации save на запись в БД
    # Пока предполагается, что векторизатор не сохраняется, но, если нужно, и его можно сохранять в save
    mm.save()