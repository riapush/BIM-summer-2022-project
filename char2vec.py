import pandas as pd
import string
from sklearn.decomposition import PCA
from operator import add

class Ch2v:
    def __init__(self):
        alphabet_l = list(string.ascii_lowercase + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        digits_l = list(string.digits)
        punctuation_l = list(string.punctuation + 'ø°№«»')
        ## one-hot encoding
        alphabet = pd.DataFrame(alphabet_l)
        digits = pd.DataFrame(digits_l)
        punctuation = pd.DataFrame(punctuation_l)
        alphabet = pd.get_dummies(alphabet).T
        digits = pd.get_dummies(digits).T
        punctuation = pd.get_dummies(punctuation).T

        ## Снижение размерности независимо для каждой группы
        pc_alph = pd.DataFrame(PCA(n_components=12).fit_transform(alphabet)).T
        pc_alph.columns = alphabet_l
        pc_dig = pd.DataFrame(PCA(n_components=2).fit_transform(digits)).T
        pc_dig.columns = digits_l
        pc_punct = pd.DataFrame(PCA(n_components=6).fit_transform(punctuation)).T
        pc_punct.columns = punctuation_l

        ## Объединение частей векторов, соответствуюхих разным группам, в один вектор
        ch2v = pd.concat([pc_alph, pc_dig, pc_punct])
        ch2v = ch2v.fillna(0)
        self.ch2v = ch2v

    def encode(self, s):
        if type(s) != str:
            s = str(s)
        chars = list(s.replace(' ', '_').lower())
        
        ## Простое кодирование по принципу сложения векторов
        vec = [0 for x in range(20)]
        for ch in chars:
            if ch in self.ch2v.columns:
                vec = list(map(add, vec, self.ch2v[ch]))
        
        return vec