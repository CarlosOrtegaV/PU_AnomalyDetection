# -*- coding: utf-8 -*-
"""
Created on Sat May 23 04:19:06 2020

@author: orteg
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

class Spy(BaseEstimator, ClassifierMixin):
  
  def __init__(self,
               spy_ratio = 0.15,
               random_state = 123):
    
    self.spy_ratio = spy_ratio
    self.random_state = random_state
  

  def fit(self, X, y):
    
    self.ix_P_ = np.where(y == 1)[0]
    self.ix_N_ = np.where(y == 0)[0]
    self.n_spy_ = round(len(self.ix_P_) * self.spy_ratio)
    self.base_classifier_ = GaussianNB()
    self.ix_spy_ = np.random.RandomState(self.random_state).choice(self.ix_P_,
                                                                  self.n_spy_,
                                                                  replace = False)
    self.y_spy = y.copy()
    self.y_spy[self.ix_spy_] = 0

    self.base_classifier_ = GaussianNB()
    self.base_classifier_.fit(X, self.y_spy)
    
    self.spy_threshold_ = np.quantile(-1 * self.base_classifier_.predict_proba(X[self.ix_spy_]), 
                                      q = 1 - self.spy_ratio)
    
    return self
  
  
  def score_samples(self, X):
    
    return -1 * self.base_classifier_.predict_proba(X)[:,1]