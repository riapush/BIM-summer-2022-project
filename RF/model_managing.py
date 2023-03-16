import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib as jl
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from text_preprocessing import TextPreprocessor

class ModelManager:
    '''
    Класс для управления моделью.
    @methods:
        : optimize_hyper - подбор гиперпараметров;
        : calibrate - калибровка модели;
        : evaluate - оценка качества модели;
        : train - обучение модели;
        : predict - предсказание по полученным данным;
        : choose_best - выбор лучшей модели;
        : save - сохранение модели;
        : load - загрузка ранее сохраненной модели;
    '''

    def __init__(self,
            model = None,
            encoder: TextPreprocessor = None):
        '''
        Инициализация менеджера моделью.
        @param self:
            : экземпляр класса;
        @param model:
            : модель;
        @param encoder:
            : кодировщик для работы с текстовыми признаками (должен быть инициализирован);
        '''
        if model == None:
            self.model = RandomForestClassifier(n_estimators=491, min_samples_split=3, min_samples_leaf=1,
                max_features='sqrt', max_depth=52, bootstrap=False)
        else:
            self.model = model

        if encoder == None:
            self.encoder = TextPreprocessor()
            self.encoder.init_ch2v()
        else:
            self.encoder = encoder


    def optimize_hyper(self,
            data: pd.DataFrame|list|pd.Series,
            classes: list|pd.Series,
            grid: dict = None,
            n_iter: int = 100,
            cv: int = 3):
        '''
        Подборгиперпараметров модели. Работает с уже векторизованными данными.
        Подразумевается, что если данные имеют истинные метки классов, значит они из набора,
        добавленного пользователями, а там векторизация производится сразу.
        @param self:
            : экземпляр класса;
        @param data:
            : данные для обучения;
        @param classes:
            : истинные классы для полученных данных;
        @param n_iter:
            : число вариантов которые будут рассмотрены при подборе гиперпараметров;
        @param params_grid:
            : сетка по которой будет происходить подбор гиперпараметров
        @param cv:
            : число срезов для кроссвалидации;
        @return:
            : возвращает модель до подбора гиперпараметров, новая модель сохраняется в менеджере;
        '''
        if isinstance(data, pd.Series):
            data = list(data)
        if grid == None:
            # Число деревьев
            n_estimators = [int(x) for x in np.linspace(start=100, stop=800, num=35)]
            # Количество признаков, которые необходимо учитывать при каждом разделении
            max_features = ['sqrt', 'log2']
            # Максимальное количество уровней в дереве
            max_depth = [int(x) for x in np.linspace(10, 110, num=22)]
            max_depth.append(None)
            # Минимальное количество выборок, необходимых для разделения узла
            min_samples_split = [x for x in range(2,8)]
            # Минимальное количество выборок, требуемых для каждого конечного узла
            min_samples_leaf = [1, 2, 3, 4]
            # Способ отбора выборок для обучения каждого дерева
            bootstrap = [True, False]
            # Создание сетки
            grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        
        # Производим подбор гиперпараметров
        model_random = RandomizedSearchCV(estimator=self.model, param_distributions=grid, n_iter=n_iter, cv=cv,
            verbose=2, n_jobs=1)
        model_random.fit(data, classes)
        
        # Сохраняем результат
        old_model = self.model
        self.model = model_random.best_estimator_
        return old_model


    def calibrate(self,
            data: pd.DataFrame|list|pd.Series,
            classes: list|pd.Series,
            cv: int = 3):
        '''
        Калибровка модели. Работает с уже векторизованными данными.
        Подразумевается, что если данные имеют истинные метки классов, значит они из набора,
        добавленного пользователями, а там векторизация производится сразу.
        @param self:
            : экземпляр класса;
        @param data:
            : данные для обучения;
        @param classes:
            : истинные классы для полученных данных;
        @param cv:
            : число срезов для кроссвалидации;
        @return:
            : возвращает модель до калибровки, новая модель сохраняется в менеджере;
        '''
        if isinstance(data, pd.Series):
            data = list(data)
        # Калибровка модели, чтобы результат predict_proba был более корректным
        calibrated_model = CalibratedClassifierCV(self.model, cv=cv)
        calibrated_model.fit(data, classes)

        # Сохраняем результат
        old_model = self.model
        self.model = calibrated_model
        return old_model


    def evaluate(self,
            data: pd.DataFrame|list|pd.Series,
            labels: list|pd.Series) -> dict:
        '''
        Оценка качества модели. Работает с уже векторизованными данными.
        Подразумевается, что если данные имеют истинные метки классов, значит они из набора,
        добавленного пользователями, а там векторизация производится сразу.
        @param self:
            : экземпляр класса;
        @param data:
            : данные для классификации;
        @param labels:
            : метки классов для полученных данных;
        @return -> dict:
            : словарь с результатами;
        '''
        if isinstance(data, pd.Series):
            data = list(data)
        res = {}
        prediction = self.model.predict(data)
        res['accuracy'] = accuracy_score(labels, prediction)
        res['f1-score(macro)'] = f1_score(labels, prediction, average='macro')

        return res


    def train(self,
            data: pd.DataFrame|list|pd.Series,
            classes: list|pd.Series):
        '''
        Тренировка модели. Работает с уже векторизованными данными.
        Подразумевается, что если данные имеют истинные метки классов, значит они из набора,
        добавленного пользователями, а там векторизация производится сразу.
        @param self:
            : экземпляр класса;
        @param data:
            : данные для обучения;
        @param classes:
            : истинные классы для полученных данных;
        @return:
            : возвращает модель до дообучения, новая модель сохраняется в менеджере;
        '''
        if isinstance(data, pd.Series):
            data = list(data)
        old_model = self.model
        self.model.fit(data, classes)
        return old_model


    def predict(self,
            df: pd.DataFrame) -> dict:
        '''
        Предсказание класса по полученным данным.
        @param self:
            : экземпляр класса;
        @param df:
            : датафрем с данными для классификации;
        @return -> dict:
            : возвращает результаты предсказаний;
        '''
        # Приводим датафрем к тому виду, который можно подать в модель
        # Здесь никаких сохранений промежуточной информации (векторизации), в отличие от имеющейся реализации, нет,
        # поскольку для поиска в БД нужны были бы очищенные значения ячеек (а это половина времени векторизации),
        # к тому же поиск по БД занимает время, так что на данном этапе выигрыш (если он все же и будет) сомнительный.
        # Вызов encode_text занимает 0.00025с в среднем, на кодирование строки датафрейма уйдет 0.003с в среднем.
        new_df = df.copy()
        new_df = self.encoder.encode_df(new_df)
        
        results = pd.DataFrame()

        # Предсказываем классы и их вероятности
        probas = self.model.predict_proba(new_df)
        results['class'] = self.model.classes_[np.argmax(probas, axis=1)]
        results['proba'] = [100 * round(x, 4) for x in probas.max(axis=1)]

        return results


    def choose_best(self,
            model,
            data: pd.DataFrame|list|pd.Series,
            classes: list|pd.Series):
        '''
        Выбор лучшей из двух моделей: полученной и сохраненной в менеджере. Работает с уже векторизованными данными.
        Подразумевается, что если данные имеют истинные метки классов, значит они из набора,
        добавленного пользователями, а там векторизация производится сразу.
        @param self:
            : экземпляр класса;
        @param model:
            : модель, с которой будет проводиться сравнение;
        @param data:
            : данные для обучения;
        @param classes:
            : истинные классы для полученных данных;
        '''
        if isinstance(data, pd.Series):
            data = list(data)
        try:
            prediction = model.predict(data)
            res = f1_score(classes, prediction, average='macro')
        except:
            return

        prediction = self.model.predict(data)
        res2 = f1_score(classes, prediction, average='macro')

        if res > res2:
            self.model = model


    # Предполагается, что будет заменен
    def save(self,
            filename: str = 'model.joblib'):
        '''
        Сохранение модели для дальнейшего использования.
        @param self:
            : экземпляр класса;
        @param filename:
            : имя выходного файла;
        '''
        jl.dump(self.model, filename)


    # Предполагается, что будет заменен
    def load(self,
            filename: str = 'model.joblib'):
        '''
        Загрузка ранее сохраненной модели.
        @param self:
            : экземпляр класса;
        @param filename:
            : имя входного файла;
        '''
        self.model = jl.load(filename)