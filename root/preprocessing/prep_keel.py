# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:04:49 2020

@author: orteg
"""
# Standard
import numpy as np
import pandas as pd

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

pg0 = pd.read_csv('page-blocks0.csv', header = None)
pk8 = pd.read_csv('poker-8-9_vs_5.csv', header = None)
ab1 = pd.read_csv('abalone19.csv', header = None)
wi4 = pd.read_csv('winequality-red-4.csv', header = None)
se0 = pd.read_csv('segment0.csv', header = None)

wilt = pd.read_csv('wilt.csv')
pizz = pd.read_csv('pizzacutter1.csv')
pie = pd.read_csv('piechart2.csv')
sat = pd.read_csv('satellite.csv')

tuple_keel = [(pg0, 'scaled_page-blocks0.csv'), 
              (pk8, 'scaled_poker-8-9_vs_5.csv'), 
              (ab1.iloc[:,1:], 'scaled_abalone19.csv'), 
              (wi4, "scaled_winequality-red-4.csv"),
              (se0, "scaled_segment0.csv")
              ]

#### PREPROCESSING ####

for df, filename in tuple_keel:
  
  X, y = df.iloc[:,:-1], df.iloc[:,-1].str.strip()

  y = np.where(y == 'negative', 0, 1)
  if len(X) > 5000:
    rs = StratifiedShuffleSplit(n_splits = 1, train_size = 5000, random_state = 123)
    ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
    X = X.iloc[ix_tr]
    y = y[ix_tr]
    
  scaler = StandardScaler()
  X_trns = scaler.fit_transform(X)
  df_trns = pd.DataFrame(np.hstack((X_trns,y.reshape(-1,1))))
  df_trns.to_csv(filename, index = False,  header = False)

  df_trns.to_csv(filename, index = False,  header = False)
  
## Wilt Preprocessing ##
X, y = wilt.iloc[:,:-1], wilt.iloc[:,-1]
y = np.where(y == 2, 1, 0)
if len(X) > 5000:
  rs = StratifiedShuffleSplit(n_splits = 1, train_size = 5000, random_state = 123)
  ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
  X = X.iloc[ix_tr]
  y = y[ix_tr]
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df_trns = pd.DataFrame(np.hstack((X_trns,y.reshape(-1,1))))
df_trns.to_csv('scaled_wilt.csv', index = False,  header = False)

## Pizza Cutter Preprocessing ##
X, y = pizz.iloc[:,:-1], pizz.iloc[:,-1]
y = np.where(y == 'N', 0, 1)
if len(X) > 5000:
  rs = StratifiedShuffleSplit(n_splits = 1, train_size = 5000, random_state = 123)
  ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
  X = X.iloc[ix_tr]
  y = y[ix_tr]
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df_trns = pd.DataFrame(np.hstack((X_trns,y.reshape(-1,1))))
df_trns.to_csv('scaled_pizzacutter1.csv', index = False,  header = False)

## Pie Chart 2 Preprocessing ##
X, y = pie.iloc[:,:-1], pie.iloc[:,-1]
y = np.where(y == 'N', 0, 1)
if len(X) > 5000:
  rs = StratifiedShuffleSplit(n_splits = 1, train_size = 5000, random_state = 123)
  ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
  X = X.iloc[ix_tr]
  y = y[ix_tr]
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df_trns = pd.DataFrame(np.hstack((X_trns,y.reshape(-1,1))))
df_trns.to_csv('scaled_piechart2.csv', index = False,  header = False)

## Satellite Preprocessing ##
X, y = sat.iloc[:,:-1], sat.iloc[:,-1]
y = np.where(y == "'Anomaly'", 1, 0)
if len(X) > 5000:
  rs = StratifiedShuffleSplit(n_splits = 1, train_size = 5000, random_state = 123)
  ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
  X = X.iloc[ix_tr]
  y = y[ix_tr]
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df_trns = pd.DataFrame(np.hstack((X_trns,y.reshape(-1,1))))
df_trns.to_csv('scaled_satellite.csv', index = False,  header = False)