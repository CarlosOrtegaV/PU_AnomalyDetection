# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:04:49 2020

@author: orteg
"""

# Standard
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#### PREPRO ####

### Loading and Filtering Data

k1 = pd.read_csv('k1.csv')
th = pd.read_csv('thyroid.csv', header = None)
wi = pd.read_csv('wine.csv', header = None)
pi = pd.read_csv('pima.csv', header = None)
co = pd.read_csv('cover.csv', header = None)
ca = pd.read_csv('cardio.csv', header = None)
f1 = pd.read_csv('f1.csv')

#### STANDARDIZE ODDS DATASETS ####
#  http://odds.cs.stonybrook.edu/
#  https://www.openml.org/d/1467
#  https://sci2s.ugr.es/keel/imbalanced.php
tuple_ = [
          (th, 'thyroid_scaled.csv'), 
          (wi, 'wine_scaled.csv'), 
          (pi, 'pima_scaled.csv'), 
          (co, "cover_scaled.csv"), 
          (ca, 'cardio_scaled.csv'),
          (cl, 'climate_scaled.csv'),
          (ys, 'yeast_scaled.csv')
          ]

for df, filename in tuple_:
  X = df.iloc[:,:-1]
  y = np.array(df.iloc[:,-1]).reshape(-1,1)

  scaler = StandardScaler()
  X_trns = scaler.fit_transform(df.iloc[:, :-1])
  df_trns = pd.DataFrame(np.hstack((X_trns, y)))
  df_trns.to_csv(filename, index = False,  header = False)
