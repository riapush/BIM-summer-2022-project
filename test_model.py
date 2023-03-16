import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best

def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs

df = pd.read_csv('./marketdata2.csv').drop(['Unnamed: 0'], axis=1)

qt = QuantileTransformer(output_distribution='normal')
df['nAveSpend'] = qt.fit_transform(df[['aveSpend']].values.reshape(-1,1))
df['nIncome'] = qt.fit_transform(df[['income']].values.reshape(-1,1))
df['nAge'] = qt.fit_transform(df[['age']].values.reshape(-1,1))
df['recent_touchpoint'].unique()
#create a mapping from labels to a unique integer and vice versa for labelling and prediction later
labels = df['recent_touchpoint'].unique()
i = 0
idx2class = {}
class2idx = {}
for tp in labels:
    idx2class[i] = tp
    class2idx[tp] = i
    i += 1
df['label'] = df['recent_touchpoint'].replace(class2idx)

train_idxs, val_idxs = stratified_split(df, 'label', val_percent=0.25)
val_idxs, test_idxs = stratified_split(df[df.index.isin(val_idxs)], 'label', val_percent=0.5)

train_df = df[df.index.isin(train_idxs)]
X_train = train_df[['nTouchpoints', 'single', 'divorced', 'married', 'unknown', 'P4',
       'P3', 'P2', 'P1', 'U', 'N', 'Y', 'C', 'B', 'D', 'A', 'F', 'E', 'New',
       'G', 'nAveSpend', 'nIncome', 'nAge']].values
Y_train = train_df[['label']].values
val_df = df[df.index.isin(val_idxs)]
X_val = val_df[['nTouchpoints', 'single', 'divorced', 'married', 'unknown', 'P4',
       'P3', 'P2', 'P1', 'U', 'N', 'Y', 'C', 'B', 'D', 'A', 'F', 'E', 'New',
       'G', 'nAveSpend', 'nIncome', 'nAge']].values
Y_val = val_df[['label']].values
test_df = df[df.index.isin(test_idxs)]
X_test = test_df[['nTouchpoints', 'single', 'divorced', 'married', 'unknown', 'P4',
       'P3', 'P2', 'P1', 'U', 'N', 'Y', 'C', 'B', 'D', 'A', 'F', 'E', 'New',
       'G', 'nAveSpend', 'nIncome', 'nAge']].values
Y_test = test_df[['label']].values

training_data = {'X_train':X_train,'Y_train':Y_train,
                'X_val': X_val,'Y_val':Y_val,
                'X_test': X_test,'Y_test': Y_test}

# clf = RandomForestClassifier(n_jobs=None, random_state=27, verbose=1)
# clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
#
# predicted_labels = clf.predict(training_data['X_test'])
#
# print(accuracy_score(training_data['Y_test'], predicted_labels))
#
# params = {
#     'n_estimators': range(450, 700, 50),
#     'max_depth': [8, 9, 10, 11, 12],
#     'max_features': ['auto'],
#     'criterion':['gini']
# }
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
# gsearch1 = GridSearchCV(estimator = clf, param_grid = params, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
# gsearch1.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
#
# clf2 = gsearch1.best_estimator_
#
# params1 = {
#     'n_estimators'      : range(400,500,10),
#     'max_depth'         : [11, 12, 13]
# }
# #metrics to consider: f1_micro, f1_macro, roc_auc_ovr
# gsearch2 = GridSearchCV(estimator=clf2, param_grid=params1, scoring='f1_micro', n_jobs=-1, verbose=10, cv=5)
# gsearch2.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
#
# print(getTrainScores(gsearch2))
#
# clf3 = gsearch2.best_estimator_
#
# params2 = {
#     'n_estimators'      : [392,],
#     'max_depth'         : [24, 25, 26]
# }
# #metrics to consider: f1_micro, f1_macro, roc_auc_ovr
# gsearch3 = GridSearchCV(estimator = clf3, param_grid = params2, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
# gsearch3.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
#
# print(getTrainScores(gsearch3))
#
# final_clf = gsearch3.best_estimator_
# final_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
# predicted_labels = final_clf.predict(training_data['X_test'])
# train_pred = final_clf.predict(training_data['X_train'])
# print('Train Accuracy:'+str(accuracy_score(training_data['Y_train'], train_pred)))
# print('Train F1-Score(Micro):'+str(f1_score(training_data['Y_train'], train_pred,average='micro')))
# print('------')
# print('Test Accuracy:'+str(accuracy_score(training_data['Y_test'], predicted_labels)))
# print('Test F1-Score(Micro):'+str(f1_score(training_data['Y_test'], predicted_labels,average='micro')))

import xgboost as xgb
import matplotlib.pyplot as plt


def plot_compare(metrics, eval_results, epochs):
    for m in metrics:
        test_score = eval_results['val'][m]
        train_score = eval_results['train'][m]
        rang = range(0, epochs)
        plt.rcParams["figure.figsize"] = [6, 6]
        plt.plot(rang, test_score, "c", label="Val")
        plt.plot(rang, train_score, "orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()


def fitXgb(sk_model, training_data=training_data, epochs=300):
    print('Fitting model...')
    sk_model.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0], ))
    print('Fitting done!')
    train = xgb.DMatrix(training_data['X_train'], label=training_data['Y_train'])
    val = xgb.DMatrix(training_data['X_val'], label=training_data['Y_val'])
    params = sk_model.get_xgb_params()
    metrics = ['mlogloss', 'merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(val, 'val'), (train, 'train')]
    xgb_model = xgb.train(params, train, epochs, evallist, evals_result=store, verbose_eval=100)
    print('-- Model Report --')
    print(
        'XGBoost Accuracy: ' + str(accuracy_score(sk_model.predict(training_data['X_test']), training_data['Y_test'])))
    print('XGBoost F1-Score (Micro): ' + str(
        f1_score(sk_model.predict(training_data['X_test']), training_data['Y_test'], average='micro')))

from xgboost.sklearn import XGBClassifier
#initial model
xgb1 = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)

fitXgb(xgb1, training_data)


