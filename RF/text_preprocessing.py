import pandas as pd
import string
from sklearn.decomposition import PCA
import re
from collections import Counter
import numpy as np

class TextPreprocessor:
    '''
    Класс для предварительной обработки текстовых данных.
    Содержит методы для очистки текстовых данных и для встраивания на уровне символов.
    @methods:
        : init_ch2v - инициализация таблицы встраивания;
        : clear_text - очистка текста;
        : encode_text - встраивание на уровне символов для строки;
        : encode_df - встраивание на уровне символов для элементов датафрейма, каждая строка датафрейма
        переводится в один элемент списка, содержащий вектор встраивания всех признаков;
    '''

    def __init__(self,
            filler: str = 'неизвестно'):
        '''
        Инициализирует слово-наполнитель для заполнения пустых значений.
        @param self:
            : экземпляр класса;
        @param filler:
            : слово-наполнитель;
        '''
        self.filler = filler


    def init_ch2v(self,
            groups: list[str] = [
                string.ascii_lowercase + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                string.digits,
                string.punctuation + 'ø°№–'
                ],
            group_vec_sizes: list[int] = [
                12, 
                2, 
                6
                ]):
        '''
        Метод, инициализирующий таблицу для встраивания на уровне символов.

        @param self:
            : экземпляр класса; 
        @param groups:
            : список строк - групп для встраивания, каждой группе выделяется часть вектора, размер которой задается
            в group_vec_sizes;
            : в результирующем векторе части, соответствующие группам, сохраняют тот же порядок,
            что был у исходных групп;
            : регистр игнорируется;
            : не имеет смысла, если установлен from_file;
        @param group_vec_sizes:
            : список размеров частей вектора встраивания для различных групп;
            : должен иметь размер, совпадающий с размером groups;
            : результирующий вектор встраивания имеет размерность равную сумме размерностей всех частей;
            : не имеет смысла, если установлен from_file;

        @warning:
            : построение встраивания для пробельных символов не поддерживается;
            : в случае возникновения ошибки бросает исключение (состояние не меняется):
                : ValueError - при некорректных значениях;
        '''
        # Проверяем параметры
        if len(groups) != len(group_vec_sizes):
            raise ValueError('groups и group_vec_sizes должны иметь одинаковый размер')
        
        # Проверяем, что размеры допустимые
        for size in group_vec_sizes:
            if size <= 0:
                raise ValueError('значения в group_vec_sizes должны быть положительными')

        # Создание списков символов для встраивания
        groups_s = [sorted(set(x.lower())) for x in groups]

        # One-hot encoding
        dfs = [pd.DataFrame(x) for x in groups_s]
        dfs = [pd.get_dummies(x).T for x in dfs]

        # Снижение размерности независимо для каждой группы
        pca_dfs = []
        for i in range(len(groups)):
            pca_df = pd.DataFrame(PCA(n_components=group_vec_sizes[i], svd_solver='full', iterated_power=2).fit_transform(dfs[i])).T
            pca_df.columns = groups_s[i]
            pca_dfs.append(pca_df)

        # Объединение частей векторов, соответствуюхих разным группам, в один вектор
        ch2v = pd.concat(pca_dfs)
        
        self.ch2v = ch2v.fillna(0.0)
        
        self.size = self.ch2v.shape[0]
        tmp = {}
        for col in self.ch2v.columns:
            tmp[col] = np.array(self.ch2v[col])
        self.ch2v = tmp


    def clear_text(self,
            text: str) -> str:
        '''
        Метод производящий очистку текста.
        Работает с удалением бессмысленных слов, заменяет '-,.' при применении не для чисел на '_',
        заменяет пробельные символы на '_', удаляет индексацию и убирает лишние нижние подчеркивания.
        @param self:
            : экземпляр класса;
        @param text:
            : текст для очистки;
        @return -> str:
            : очищенный текст;
        '''
        # Заменяем некоторые разделители на нижнее подчеркивание
        # "Повышаем" значимость -,. как чего-то, характерного числам
        text = re.sub(r'((?<!\d)(\.|,))|((\.|,|(-|–))(?!\d))|((?<!(_|\s))(-|–)(?=\d))', '_', text)

        # Заменяем пробельные символы на нижнее подчеркивание
        text = re.sub(r'\s', '_', text) 

        # Приводим текст к нижнему регистру, удаляем ненужные слова
        # Для слов рассмотрены варианты замены латиницы на кирилицу и наоборот в отдельных символах
        # (подобное встречалось в исходных данных)
        text = re.sub(r'(_*<*_*(((a|а)ds(k|к))|(тип)|(г(о|o)ст)|(т(у|y))|(n(o|о)n(e|е))|(unn(a|а)m(e|е)d)|\
            ((o|о)th(e|е)r)|(n(o|о)td(e|е)fin(e|е)d)|(б(e|е)з_*им(e|е)ни)|(\!н(e|е)_*(у|y)читыв(a|а)ть))_*>*_*)',
            '', text.lower())
        
        # Удаляем индексы из строк
        text = re.sub(r'(_*:_*\d{6,10})|(_*\(_*(\d+|\?)_*\)$)', '', text)

        # Удаляем повторы нижних подчеркиваний
        text = re.sub(r'(_)\1+', r'\1', text)
        
        # Удаляем нижнее подчеркивание в начале и конце
        text = re.sub(r'^_|_$', '', text)

        # Если от текста ничего не осталось используем наполнитель
        if text == '':
            text = self.filler

        return text


    def encode_text(self,
            text: str|int|float) -> np.array:
        '''
        Метод непосредственно осуществляющий встраивание на уровне символов.
        Символы, не входящие в таблицу встраивания игнорируются. Использует clear_text.
        @param self:
            : экземпляр класса;
        @param text:
            : строка (или число, которое будет приведено к строке), для которой требуется встраивание;
        @return -> np.array:
            : вектор встраивания, размерность которого была задана при инициализации таблицы встраивания;

        @warning:
            : в случае ошибки кидает AttributeError;
        '''
        # Приведедние типа объекта к строке
        if type(text) != str:
            text = str(text)
        # Очистка текста
        else:
            text = self.clear_text(text)
        
        # Простое кодирование по принципу сложения векторов
        try:
            vec = np.array([0 for x in range(self.size)])
            counter = Counter(text)
            for ch in counter:
                if ch in self.ch2v:
                    vec = vec + counter[ch] * self.ch2v[ch]
        except:
            raise AttributeError('таблица встраивания не инициализирована')
        
        return vec


    def encode_df(self,
            df: pd.DataFrame) -> list:
        '''
        Метод, осуществляющий встраивание для каждого элемента датафрейма. Использует encode_text.
        @param self:
            : экземпляр класса;
        @param df:
            : датафрейм, для которого производится встраивание;
        @return -> list:
            : набор векторов встраивания большой размерности, соответсвующих строкам датарейма;
        '''
        # Заполним пропуски в исходном датафрейме
        tmp_df = df.fillna(self.filler, inplace=False)
        tmp_df.columns = [str(x) for x in df.columns]
        order = sorted(tmp_df.columns)

        data = [[] for i in range(tmp_df.shape[0])]
        res_ser = pd.Series(data, tmp_df.index)

        # Производим встраивание
        for col in order:
            emb = tmp_df[col].apply(self.encode_text)
            for ind in res_ser.index:
                for i in range(self.size):
                    res_ser[ind].append(emb[ind][i])
        
        return list(res_ser)