import data_layer_aia as data_layer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
from talib import abstract
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from datetime import timedelta
import time
import random
import os
import json

def data_cleaning(df):
    # fill na
    df.interpolate(method='linear', limit_direction='forward', axis=0 ,inplace=True)
    
    #remove duplicate rows with duplicate index or value
    if df.index.duplicated().any() or df.duplicated().any():
        mask = df.index[df.index.duplicated()]
        df.drop(mask,inplace=True)
        mask = df[df.duplicated()].index
        df.drop(mask,inplace=True)
    return df

### Get HSI data as feature and conduct df clearning and matching ###
def get_hsi(df):
    end = df.index[-1]+ timedelta(1)
    hsi_df = yf.download("^HSI", start=df.index[0], end=end)
    hsi_df_c = data_cleaning(hsi_df)
    hsi_close_c = hsi_df_c['Adj Close']
    hsi_close_c = hsi_close_c.rename('hsi_close')
    ### cleaning for df ###
    # 1. Holiday cleaning 2. Fill na inside the dense data
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    holiday_index = [index for index in df.index if index not in hsi_df_c.index]
    df = df.drop(holiday_index)
    merge_df = pd.merge(df, hsi_close_c, left_index=True, right_index=True)
    return merge_df

def feature_engineer(df):
    df = df.astype('float')
    df.columns = ['close','hsi_close']
    df = df.dropna()
    ta_list = ['BBANDS(df,20)','EMA(df,50)','EMA(df,200)',
              'MACD(df)','RSI(df)','LINEARREG_SLOPE(df,10)','TAN(df)']
    for x in ta_list:
        try:
            output = eval('abstract.'+x)
            output.name = x.lower() if type(output) == pd.core.series.Series else None
            df = pd.merge(df, pd.DataFrame(output), left_on = df.index, right_on = output.index)
            df = df.set_index('key_0')
        except:
            print(x)
    df['golden cross'] = np.where(df['ema(df,50)']>df['ema(df,200)'],1,0)
    df['price > ma50'] = np.where(df['close']>df['ema(df,50)'],1,0)
    df['macd > 0'] = np.where(df['macd']>0,1,0)
    df['price > upperband'] = np.where(df['close']>df['upperband'],1,0)
    df['price > middleband'] = np.where(df['close']>df['middleband'],1,0)
    df['price > lowerband'] = np.where(df['close']>df['lowerband'],1,0)
    df.dropna(inplace=True)
    target = df['close'].pct_change(30).shift(-30) # return of one month
    df['target'] = np.where(target>0,1,0)
    # df.index.name = 'Date'
    df.drop(['upperband', 'middleband', 'lowerband','ema(df,50)', 'ema(df,200)'],axis=1,inplace=True)
    return df

def train_test_split(df, train_ratio):
    split = int(len(df)*train_ratio)
    X_train = df[:split].iloc[:,:-1]
    y_train = df[:split].iloc[:,-1]
    X_test = df[split-30:].iloc[:,:-1]
    y_test = df[split-30:].iloc[:,-1]
    return X_train, y_train, X_test, y_test

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def gen_logistic_regression(X_train, y_train, n_split=5):
    model = LogisticRegression(solver="liblinear")
    lr_model = Pipeline([('scaler', StandardScaler()), ('model', model)])
    # evaluate the model
    btscv = BlockingTimeSeriesSplit(n_splits=n_split)
    scores = cross_validate(lr_model, X_train, y_train, scoring=['accuracy', 'precision'], cv=btscv, n_jobs=-1)
    lr_model.fit(X_train,y_train)
    return lr_model

def gen_SVM(X_train, y_train, n_split=5):
    strategy = ['linear','rbf','sigmoid']
    for str_item in strategy:
        # define the reference model
        model = SVC(kernel=str_item)
        svc_model = Pipeline([('scaler', StandardScaler()), ('model', model)])
        # evaluate the model
        btscv = BlockingTimeSeriesSplit(n_splits=n_split)
        scores = cross_validate(svc_model, X_train, y_train, scoring=['accuracy', 'precision'], cv=btscv, n_jobs=-1)
        svc_model.fit(X_train,y_train)
    return svc_model

def gen_binary_lightgbm(X_train, y_train, round_num, metric):
    params = {
#     'learning_rate' : [round(x,2) for x in np.random.uniform(0, 1, 10)],
    'model__objective':['binary'],
    'model__learning_rate' : [0.001,0.01,0.05,0.08,0.1],
#     'model__max_depth' : np.random.randint(5, 20,5),
    'model__feature_fraction' : [0.5,0.7,0.8,0.9],
#     'num_leaves': np.random.randint(1, 10,5),
    'model__min_data_in_leaf': np.random.randint(5, 10,5),
#     'lambda_l1': [round(x,2) for x in np.random.uniform(0, 1, 3)],
#     'model__lambda_l2': np.random.randint(0, 0.5 ,3),
    'model__lambda_l2': [0.005,0.05,0.1],
    'model__boosting_type' : ['gbdt'],
    'model__reg_alpha' : [0,0.05,0.5,1],
    'model__reg_lambda' : [0,0.05,0.5,1]
    }
    # model__fit_params={'model__early_stopping_rounds':20,'model__eval_set':[(X_train,y_train)]}
    # btscv = BlockingTimeSeriesSplit(n_splits=5)
    gkf = KFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train)
    lgb_estimator = lgb.LGBMClassifier(objective='binary', num_boost_round=round_num,scale_pos_weight = 0.5)
    
    gsearch = GridSearchCV(
        estimator = Pipeline([('scaler', StandardScaler()), ('model', lgb_estimator)]),
        param_grid=params,
        n_jobs=-1,
        scoring = metric,
        refit= metric,
        cv=gkf,  # customerised splitter subject to test
        verbose=-1,
        pre_dispatch=8,
        error_score=-999,
        return_train_score=True
        )

    lgb_model = gsearch.fit(X_train, y_train)
    return lgb_model

def get_prediction(X_test, models, weight):
    finalpred = 0
    for m, w in zip(models, weight):
        prediction = m.predict(X_test)
        finalpred += prediction * w

    #convert into binary values
    for i in range(len(finalpred)):
        if finalpred[i] >= 0.7:
            finalpred[i] = 1
        else:
            finalpred[i] = 0
    return finalpred

def Generate_weightings(selected_df):
#     df -> array
    portfolio_weightings = []
    # Iterate over column names 
    for column in selected_df:
        # Select column contents by column 
        # name using [] operator 
        columnSeriesObj = selected_df[column]
        if np.isnan(columnSeriesObj[row]) or (column not in selected_fund and len(selected_fund) > 0):
            weight = 0
        else:
            weight = np.random.random_sample()
        portfolio_weightings.append(weight)

    #below line ensures that the sum of our weights is 1
    sum_of_weightings = sum(portfolio_weightings)
    portfolio_weightings = [x/sum_of_weightings for x in portfolio_weightings]

#         weight redistribution implement: for weights < 0.02 -> redistribute it to the largest weight element
    ind = np.argmax(portfolio_weightings)
    residual_weights = 0
    for i, weight in enumerate(portfolio_weightings):
        if weight < 0.02:
            residual_weights += weight
            portfolio_weightings[i] = 0
    portfolio_weightings[ind] += residual_weights
    return np.array(portfolio_weightings)

def portfolios(selected_df):
#     df -> weight list, risk list

    number_of_portfolios = 500
    RF = 0.02  #According to the newest minimun ibond rate
    
    portfolio_returns = []
    portfolio_weights = []
    portfolio_risk = []
    sharpe_ratio_port = []
    
    for portfolio in range (number_of_portfolios):
        weights = Generate_weightings(selected_df)
        portfolio_weights.append(weights)
        
        # return 
        return_df = selected_df.pct_change()
        return_df.dropna(inplace = True)
        period_return = np.sum(return_df.mean() * weights)
        portfolio_returns.append(period_return)
        
        # std
        matrix_covariance_portfolio = (return_df.cov())
        matrix_covariance_portfolio.fillna(0,inplace=True)
        portfolio_variance = np.dot(weights.T,np.dot(matrix_covariance_portfolio, weights))
        portfolio_standard_deviation= np.sqrt(portfolio_variance) 
        portfolio_risk.append(portfolio_standard_deviation)
        
        #sharpe_ratio
        sharpe_ratio = ((period_return- RF)/portfolio_standard_deviation)
        sharpe_ratio_port.append(sharpe_ratio)
        
    return np.array(portfolio_weights), np.array(sharpe_ratio_port)

def select_maxsharpe(sharpe_ls):
    max_sharpe_ix = np.argmax(sharpe_ls)
    return max_sharpe_ix

def capital_allocation(row,cash_limit):
#     row -> list(fund unit), remaining balance
    fund_purchase_unit = []
    global available_capital
    for cash,val in zip(cash_limit,row):
        if np.isnan(val):
            fund_purchase_unit.append(0)
        else:
            unit = cash // val
            fund_purchase_unit.append(unit)
            available_capital += cash - (val * unit)
            available_capital -= val * unit
    return np.array(fund_purchase_unit, dtype=int)
