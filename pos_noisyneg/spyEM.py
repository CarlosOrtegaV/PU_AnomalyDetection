# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:53:49 2020

@author: orteg
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from scipy.sparse import issparse
import logging
from sklearn.naive_bayes import GaussianNB
from pomegranate.distributions import NormalDistribution
from pomegranate.NaiveBayes import NaiveBayes
from imblearn.over_sampling import ADASYN


class SpyEM(BaseEstimator, ClassifierMixin):
  """ Spy Expectation Maximization
  
  Parameters
  ----------
  spy_ratio : str or None, optional (default = 0.10)
    Percentage of positive instances turned into negatives (spies).
    
  threshold : float, optional (default = 0.15; 0.05, 0.10 and 0.20 also recommended by authors)
    Threshold represent the quantile of the conditional prob. of the spies.
    Reliable negatives are the unlabeled with lower than threshold.
  
  keep_treated : bool, optional (default = True)
  
  keep_final : bool, optional (default = True)
  
  random_state : int, optional (None)
  
  Attributes
  ----------
    
  Xt_, yt_ : training set after treatment (if keep_treated=True)
  
  Xf_, yf_ : training set after resampling (if keep_final=True)
  
  base_classifier : Gaussian Naive Bayes
    Classifier for predicting in Step 1.
    
  final_classifier : scikit-learn classifier, (default = None)
    Classifier for predicting in Step 1.
      
  """
  def __init__(self, 
               spy_ratio = 0.10, 
               threshold = 0.15, 
               keep_treated = True, 
               keep_final = True,
               resampler = True,
               random_state = None):
    self.em_classifier = None
    self.final_classifier = None
    self.spy_ratio = spy_ratio
    self.threshold = threshold
    self.keep_treated = keep_treated
    self.keep_final = keep_final
    self.resampler = resampler
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit estimator.
    
    Parameters
    ----------
    X : Array-like of shape = [n_samples, n_features]
      Input samples.
    y : Array of shape = [n_samples]
      Predicted classes.
        
    Returns
    -------
    self : object
        Fitted estimator.
    """
    self.ix_P_ = None
    self.ix_U_ = None
    self.ix_Spy_ = None
    self.ix_reliable_negs_ = None
    self.ix_unreliable_negs_ = None
    self.score_step1_ = None
    self.score_step2_ = None
    self.Xt_, self.yt_       = None, None
    self.Xf_, self.yf_       = None, None

    # Don't reconstruct these internal objects if we have already fitted before,
    # as we might have used the random state
    if not self._is_fitted():
      self.random_state_ = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
      self.base_classifier_ = GaussianNB()
      self.final_classifier_ = GaussianNB()
      self.em_classifier_  = self.em_classifier
      
    if self.resampler:
      self.resampler_ = ADASYN(sampling_strategy=1.0, random_state = self.random_state_, n_jobs = -1)    
            
    #### CHECKS ####
    
    # Sparsity checks
    if issparse(X) or issparse(y):
      logging.info('`X` or `y` are sparse, I will convert them to dense (might incur high memory usage)')
    
    self.Xt_ = np.asarray(X).copy() if not issparse(X) else X.toarray().copy()
    self.yt_ = np.asarray(y).copy() if not issparse(y) else y.toarray().copy()
    
    self.modified_instances_ = np.array([])
    self.score_samples_ = np.array([])
    
    # Binary checks
    unique_y = np.unique(self.yt_)
    if not len(unique_y.shape) == 1 or not unique_y.shape[0] == 2 or not (0 in unique_y and 1 in unique_y):
        raise ValueError("This classifier works binary 0/1 labels, yours are {}".format(unique_y))
        
    ##########################################################################
    self.ix_P_ = np.where(self.yt_ == 1)[0]
    self.ix_N_ = np.where(self.yt_ == 0)[0]
    
    n_spy = max(round(len(self.ix_P_) * self.spy_ratio), 1)

    self.ix_Spy_ =  np.random.RandomState(self.random_state_).choice(self.ix_P_,
                                                                     n_spy,
                                                                     replace = False)
    
    self.yt_[self.ix_Spy_] = 0
            

    self.base_classifier_.fit(self.Xt_, self.yt_)
    self.score_step1_ = self.base_classifier_.predict_proba(self.Xt_)[:, 1]
    
    self.threshold_ = np.quantile(self.score_step1_[self.ix_Spy_], q = self.threshold)
    reliable_negs_mask_ = self.base_classifier_.predict_proba(self.Xt_[self.ix_N_])[:,1] < self.threshold_
    
    self.ix_reliable_negs_ = self.ix_N_[reliable_negs_mask_]
    self.ix_unreliable_negs_ = self.ix_N_[~reliable_negs_mask_]
    
    self.yt_[self.ix_unreliable_negs_] = -1
  
    if self.ix_reliable_negs_.size == 0:
      self.ix_forced_reliable_negs_ = np.random.RandomState(self.random_state_).choice(self.ix_unreliable_negs_,
                                                                                       round(len(self.ix_unreliable_negs_) * self.threshold),
                                                                                       replace = False)
      self.ix_reliable_negs_ = self.ix_forced_reliable_negs_
    
    self.yt_[self.ix_reliable_negs_] = 0
    # Change back to positive the spy instances
    self.yt_[self.ix_Spy_] = 1
    
    # if self.resampler:
      
    #   self.yres_ = self.yt_[self.yt_ != -1].copy()
    #   self.Xres_ = self.Xt_[self.yt_ != -1].copy()
    
    #   self.Xres_, self.yres_ = self.resampler_.fit_resample(self.Xres_, self.yres_)
      
    #   self.Xres_ = np.concatenate((self.Xres_, self.Xt_[self.yt_ == -1]), axis = 0)
    #   self.yres_ = np.concatenate((self.yres_, self.yt_[self.yt_ == -1]), axis = 0)    
    
    #   self.em_classifier_ = NaiveBayes.from_samples(NormalDistribution, self.Xres_, self.yres_)
      
    # else:
      
    self.em_classifier_ = NaiveBayes.from_samples(NormalDistribution, self.Xt_, self.yt_)
            
    self.yt_[self.ix_N_] = self.em_classifier_.predict(self.Xt_[self.ix_N_])
    
    self.Xf_ = self.Xt_

    # if self.resampler:
    #   self.Xf_ = self.Xres_

    self.final_classifier_.fit(self.Xf_, self.yt_)
    
    self.yf_ = self.final_classifier_.predict(self.Xf_)
    
    if self.keep_treated == False:
      self.Xt_ = None
      self.yt_ = None
      
    if self.keep_final == False:
      self.Xf_ = None
      self.yf_ = None
      
    if self.keep_treated == True:
      self.yt_ = self.yt_.astype(int)
      
    if self.keep_final == True:
      self.yf_ = self.yf_.astype(int)
      
    return self
  
  def predict(self, X):
    """ Predict class for X.
  
      Parameters
      ----------
      X : array-like of shape = [n_samples, n_features]
  
      Returns
      -------
      y : array of shape = [n_samples]
          The predicted classes.
    """
    if not self._is_fitted():
      raise NotFittedError('Not fitted yet, call `fit` first')
    
    try:
      return self.final_classifier_.predict(X)
    except NotFittedError as e:
      print(repr(e))
  
  def predict_proba(self, X):
    """Predict class probabilities for X.

    The predicted class probabilities.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The input samples.

    Returns
    -------
    p : array of shape = [n_samples, n_classes]
        The class probabilities of the input samples.
    """
    if not self._is_fitted():
      raise NotFittedError('Not fitted yet, call `fit` first')
      
    try:
      if self.final_classifier_.predict_proba(X).shape[1] == 1:
        aux_predict_prob = np.ones((X.shape[0],2))
        aux_predict_prob[:,0] -= self.final_classifier_.predict_proba(X).reshape(-1,)
        return aux_predict_prob
      
      else:
        return self.final_classifier_.predict_proba(X)
        
    except NotFittedError as e:
      print(repr(e))
    
  def _is_fitted(self):
    return hasattr(self, 'final_classifier_')
      
  def get_params(self, deep=True):
    return {
      'base_classifier': self.base_classifier_,
      'final_classifier': self.final_classifier_,
      'spy_ratio': self.spy_ratio,
      'threshold': self.threshold,
      'keep_treated': self.keep_treated,
      'keep_final': self.keep_final,
      'random_state': self.random_state
    }

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self