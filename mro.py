import pandas as pd
from datetime import date
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import auc, recall_score,precision_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# time window function
def roll(df, j):            #df为每个id的time series，j为time window
    window_df = pd.DataFrame()
    window_df['mro'] = df['mro']
    for i in range(0, j):
        window_df['hard_braking_' + str(i)] = df['hard_braking'].shift(-i)
        window_df['speeding_sum_' + str(i)] = df['speeding_sum'].shift(-i)
        window_df['day_mileage_' + str(i)] = df['day_mileage'].shift(-i)
        window_df['tavg_' + str(i)] = df['tavg'].shift(-i)
        window_df['cust_part_cost_wo_core_' + str(i)] = df['cust_part_cost_wo_core'].shift(-i)

    window_df[['purchase_time','engn_size','purchaser_age_at_tm_of_purch','male', 'est_hh_incm_prmr_cd', 'gmqualty_model', 'umf_xref_finc_gbl_trim']] = df[['purchase_time','engn_size','purchaser_age_at_tm_of_purch','male', 'est_hh_incm_prmr_cd', 'gmqualty_model', 'umf_xref_finc_gbl_trim']]
    return window_df

def encoder(df):
    x = pd.get_dummies(df['purchase_time'], prefix = 'purchase_', drop_first=True)
    y = pd.get_dummies(df['gmqualty_model'], drop_first=True)
    z = pd.get_dummies(df['umf_xref_finc_gbl_trim'], drop_first=True)
    e = pd.get_dummies(df['est_hh_incm_prmr_cd'], drop_first=True)

    #去掉一列
    # purchase time和income变成dummy
    temp = df.drop(['gmqualty_model','umf_xref_finc_gbl_trim','purchase_time','est_hh_incm_prmr_cd'], axis=1)
    df_new = pd.concat([temp,x, y,z,e], axis=1)
    return df_new

def reshape(df, j):
    window_df = roll(df,j)
    window_df_new = window_df.dropna()      #比如window = 2，就把最后两行删掉
    return window_df_new

# reshape
df = pd.read_csv('ready_for_rolling.csv', index_col=0)
id_lst = list(set(df['id']))
for i in range(0,5000):
    slice = df[df['id'] == id_lst[i]]
    slice = reshape(slice, 60)
    df_new = pd.concat([df_new, slice], axis=0)
df_new = encoder(df_new)            #one-hot-encoding
df_new.to_csv('rolling_sample_60.csv')

# modeling
df_new = pd.read_csv('rolling_sample_60.csv', index_col=0)

X = df_new.iloc[:,1:]
X = X[X.columns.drop(list(X.filter(regex='cust_part_cost_wo_core')))]       #exclude cost
y = df_new['mro']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# grid search for 'scale_pos_weight' in XGBoost
recall_train = []
recall_val = []
precision_train = []
precision_val = []
scale_pos_weight_lst = []

for scale_pos_weight in [0.1, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]:
    scale_pos_weight_lst.append(scale_pos_weight)

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric=["error", "logloss"],random_state=4, scale_pos_weight = scale_pos_weight)

    #stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    score_val_sub = []
    score_tra_sub = []
    precision_train_sub = []
    precision_val_sub = []

    for train_index, val_index in skf.split(X_train, y_train): 
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index] 
        xgb_model.fit(X_train_fold, y_train_fold) 
        pred_train = xgb_model.predict(X_train_fold)
        pred_val = xgb_model.predict(X_val_fold)
        score_tra_sub.append(recall_score(np.array(y_train_fold), pred_train))
        score_val_sub.append(recall_score(np.array(y_val_fold), pred_val))
        precision_train_sub.append(precision_score(np.array(y_train_fold), pred_train))
        precision_val_sub.append(precision_score(np.array(y_val_fold), pred_val))

    # result
    recall_train.append(np.mean(score_tra_sub))
    recall_val.append(np.mean(score_val_sub))
    precision_train.append(np.mean(precision_train_sub))
    precision_val.append(np.mean(precision_val_sub))

result = pd.DataFrame()
result['scale_pos_weight'] = scale_pos_weight_lst
result['recall_train'] = recall_train
result['recall_val'] = recall_val
result['precision_train'] = precision_train
result['precision_val'] = precision_val
result.plot.line(x='scale_pos_weight', y=['recall_train', 'recall_val', 'precision_train', 'precision_val'],marker='.', figsize=(8,4))

#若根据validation set上的recall值来选择参数，则令scale_pos_weight =700，在test set上测试
xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric=["error", "logloss"],random_state=4, scale_pos_weight = 700)
xgb_model.fit(X_train, y_train)
pred = xgb_model.predict(X_test)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
recall, precision
print(confusion_matrix(y_test, pred))


# grid search for focal loss
recall_train = []
recall_val = []
precision_train = []
precision_val = []
gamma_lst = []
alpha_lst = []

# params
params = {'objective': 'binary:logistic','eval_metric':'error'}

for gamma in [1,2,3,4,5]:
    for alpha in [0.2,0.4,0.6,0.8,0.9]:

        gamma_lst.append(gamma)
        alpha_lst.append(alpha)

         #stratified k-fold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        score_val_sub = []
        score_tra_sub = []
        precision_train_sub = []
        precision_val_sub = []

        for train_index, val_index in skf.split(X_train, y_train): 
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index] 

            # dmatrix
            dtrain=xgb.DMatrix(X_train_fold, y_train_fold)
            dval=xgb.DMatrix(data=X_val_fold,label=y_val_fold)

             #focal loss
            def logistic_obj(p, dtrain):
                y = dtrain.get_label()
                p = 1.0 / (1.0 + np.exp(-p))
                grad = p * (1 - p) * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p))
                hess = p * (1 - p) * (p * (1 - p) * (-alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p)) + alpha * y * (1 - p) ** gamma / p ** 2 - gamma ** 2 * p ** gamma * ( 1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + 2 * gamma * p ** gamma * (1 - alpha) * (
                            1 - y) / (p * (1 - p)) + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2) - p * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                                  1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) * ( alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * ( 1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log( 1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)))
                return grad, hess
       
            xgb_model = xgb.train(params=params, dtrain=dtrain, evals=[(dtrain, 'train'), (dtest, 'test')], verbose_eval=0, obj=logistic_obj)

            pred_train = xgb_model.predict(dtrain)
            pred_val = xgb_model.predict(dval)
            pred_train[pred_train < 0.5] = 0
            pred_train[pred_train >= 0.5] = 1
            pred_val[pred_val < 0.5] = 0
            pred_val[pred_val >= 0.5] = 1

            # result
            score_tra_sub.append(recall_score(np.array(y_train_fold), pred_train))
            score_val_sub.append(recall_score(np.array(y_val_fold), pred_val))
            precision_train_sub.append(precision_score(np.array(y_train_fold), pred_train))
            precision_val_sub.append(precision_score(np.array(y_val_fold), pred_val))

        recall_train.append(np.mean(score_tra_sub))
        recall_val.append(np.mean(score_val_sub))
        precision_train.append(np.mean(precision_train_sub))
        precision_val.append(np.mean(precision_val_sub))

result = pd.DataFrame()
result['gamma'] = gamma_lst
result['alpha'] = alpha_lst
result['recall_train'] = recall_train
result['recall_val'] = recall_val
result['precision_train'] = precision_train
result['precision_val'] = precision_val

fig = plt.figure()  #定义新的三维坐标轴
ax1 = plt.axes(projection='3d')
ax1.plot(xs = result['gamma'], ys = result['alpha'], zs = result['recall_val'],marker=".")
plt.title('recall_val')

#若根据validation set上的recall值来选择参数，则令gamma = 1, alpha = 3，在test set上测试
dtrain=xgb.DMatrix(X_train, y_train)
dtest=xgb.DMatrix(data=X_test,label=y_test)

#focal loss
def logistic_obj(p, dtrain):
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-p))
    grad = p * (1 - p) * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p))
    hess = p * (1 - p) * (p * (1 - p) * (-alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p)) + alpha * y * (1 - p) ** gamma / p ** 2 - gamma ** 2 * p ** gamma * ( 1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + 2 * gamma * p ** gamma * (1 - alpha) * (
                            1 - y) / (p * (1 - p)) + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2) - p * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                                  1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) * ( alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * ( 1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log( 1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)))
    return grad, hess
       
xgb_model = xgb.train(params=params, dtrain=dtrain, evals=[(dtrain, 'train'), (dtest, 'test')], verbose_eval=0, obj=logistic_obj)
pred_test = xgb_model.predict(dtest)
pred_test[pred_test < 0.5] = 0
pred_test[pred_test >= 0.5] = 1

recall = recall_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)
recall, precision
print(confusion_matrix(y_test, pred_test))


# resample the training set + focal loss

#resample
mask = range(240, X_train.shape[1])
mask = np.array(mask)
mask.shape
sm = SMOTENC(random_state=42, categorical_features = mask, sampling_strategy = 1)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

#grid search
recall_train = []
recall_val = []
precision_train = []
precision_val = []
gamma_lst = []
alpha_lst = []
# params
params = {'objective': 'binary:logistic','eval_metric':'error'}

for gamma in [1,2,3,4,5]:
    for alpha in [0.2,0.4,0.6,0.8,0.9]:

        gamma_lst.append(gamma)
        alpha_lst.append(alpha)

         #stratified k-fold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        score_val_sub = []
        score_tra_sub = []
        precision_train_sub = []
        precision_val_sub = []

        for train_index, val_index in skf.split(X_train_resampled, y_train_resampled): 
            X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
            y_train_fold, y_val_fold = y_train_resampled.iloc[train_index], y_train_resampled.iloc[val_index] 

            # dmatrix
            dtrain=xgb.DMatrix(X_train_fold, y_train_fold)
            dval=xgb.DMatrix(data=X_val_fold,label=y_val_fold)

             #focal loss
            def logistic_obj(p, dtrain):
                y = dtrain.get_label()
                p = 1.0 / (1.0 + np.exp(-p))
                grad = p * (1 - p) * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p))
                hess = p * (1 - p) * (p * (1 - p) * (-alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p)) + alpha * y * (1 - p) ** gamma / p ** 2 - gamma ** 2 * p ** gamma * ( 1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + 2 * gamma * p ** gamma * (1 - alpha) * (
                            1 - y) / (p * (1 - p)) + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2) - p * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                                  1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) * ( alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * ( 1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log( 1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)))
                return grad, hess
       
            xgb_model = xgb.train(params=params, dtrain=dtrain, verbose_eval=0, obj=logistic_obj)

            pred_train = xgb_model.predict(dtrain)
            pred_val = xgb_model.predict(dval)
            pred_train[pred_train < 0.5] = 0
            pred_train[pred_train >= 0.5] = 1
            pred_val[pred_val < 0.5] = 0
            pred_val[pred_val >= 0.5] = 1

            # result
            score_tra_sub.append(recall_score(np.array(y_train_fold), pred_train))
            score_val_sub.append(recall_score(np.array(y_val_fold), pred_val))
            precision_train_sub.append(precision_score(np.array(y_train_fold), pred_train))
            precision_val_sub.append(precision_score(np.array(y_val_fold), pred_val))

        recall_train.append(np.mean(score_tra_sub))
        recall_val.append(np.mean(score_val_sub))
        precision_train.append(np.mean(precision_train_sub))
        precision_val.append(np.mean(precision_val_sub))

result = pd.DataFrame()
result['gamma'] = gamma_lst 
result['alpha'] = alpha_lst 
result['recall_train'] = recall_train
result['recall_val'] = recall_val
result['precision_train'] = precision_train
result['precision_val'] = precision_val
fig = plt.figure()  #定义新的三维坐标轴
ax1 = plt.axes(projection='3d')
ax1.plot(xs = result['gamma'], ys = result['alpha'], zs = result['recall_val'],marker=".")
plt.title('recall_val')

#若根据validation set上的recall值来选择参数，则令gamma = 1, alpha = 3，在test set上测试
dtrain=xgb.DMatrix(X_train_resampled, y_train_resampled)
dtest=xgb.DMatrix(data=X_test,label=y_test)

#focal loss
def logistic_obj(p, dtrain):
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-p))
    grad = p * (1 - p) * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p))
    hess = p * (1 - p) * (p * (1 - p) * (-alpha * gamma ** 2 * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) ** 2 + 2 * alpha * gamma * y * (1 - p) ** gamma / (p * (1 - p)) + alpha * y * (1 - p) ** gamma / p ** 2 - gamma ** 2 * p ** gamma * ( 1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + 2 * gamma * p ** gamma * (1 - alpha) * (
                            1 - y) / (p * (1 - p)) + gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(1 - p) / p ** 2 + p ** gamma * (1 - alpha) * (1 - y) / (1 - p) ** 2) - p * (alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * (1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log(
                                  1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) * ( alpha * gamma * y * (1 - p) ** gamma * np.log(p) / (1 - p) - alpha * y * ( 1 - p) ** gamma / p - gamma * p ** gamma * (1 - alpha) * (1 - y) * np.log( 1 - p) / p + p ** gamma * (1 - alpha) * (1 - y) / (1 - p)))
    return grad, hess
       
xgb_model = xgb.train(params=params, dtrain=dtrain, verbose_eval=0, obj=logistic_obj)
pred_test = xgb_model.predict(dtest)
pred_test[pred_test < 0.5] = 0
pred_test[pred_test >= 0.5] = 1

recall = recall_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)
recall, precision
print(confusion_matrix(y_test, pred_test))