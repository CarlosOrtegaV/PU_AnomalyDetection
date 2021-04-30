# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:04:49 2020

@author: orteg
"""

# Own packages, put that package in whatever PATH you use in python
from preprocesamiento.utils import CatNarrow, Columns, text_preprop
from entubacion.hyperhelper import EstimatorSelectionHelper

# Standard
import numpy as np
import pandas as pd

# sklearn
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# scipy
import scipy.io
mat = scipy.io.loadmat('vertebral.mat')

#### PREPRO ####

X, y = mat['X'], mat['y']
#ix = np.random.RandomState(123).choice(range(len(X)), size = 5000, replace = False)
scaler = StandardScaler()
X_trns = scaler.fit_transform(X)
df = pd.DataFrame(np.hstack((X_trns,y)))
df = pd.DataFrame(np.hstack((X,y)))

df.isna().any()
df.to_csv('scaled_vertebral.csv', index = False, header = False)

df2 = pd.read_csv('seismic.csv', header = None)




### Loading and Filtering Data

k1 = pd.read_csv('k1.csv')
k2 = pd.read_csv('k2.csv')
th = pd.read_csv('thyroid.csv', header = None)
wi = pd.read_csv('wine.csv', header = None)
pi = pd.read_csv('pima.csv', header = None)
co = pd.read_csv('cover.csv', header = None)
ca = pd.read_csv('cardio.csv', header = None)
f1 = pd.read_csv('f1.csv')

#### STANDARDIZE ODDS DATASETS ####
#  http://odds.cs.stonybrook.edu/

tuple_ = [(th, 'thyroid_scaled.csv'), 
          (wi, 'wine_scaled.csv'), 
          (pi, 'pima_scaled.csv'), 
          (co, "cover_scaled.csv"), 
          (ca, 'cardio_scaled.csv')
          ]

for df, filename in tuple_:
  X = df.iloc[:,:-1]
  y = np.array(df.iloc[:,-1]).reshape(-1,1)

  scaler = StandardScaler()
  X_trns = scaler.fit_transform(df.iloc[:, :-1])
  df_trns = pd.DataFrame(np.hstack((X_trns, y)))
  df_trns.to_csv(filename, index = False,  header = False)