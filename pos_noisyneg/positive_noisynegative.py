# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:15 2019

@author: orteg
"""
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
from imblearn.over_sampling import ADASYN
from pomegranate import NormalDistribution, NaiveBayes

import numpy as np

#from safeu.classification.TSVM import TSVM

from .spy import Spy
from .weighted_iforest import WeightedIsoForest
from .semiboost import SemiBoostClassifier

class PNN(BaseEstimator, ClassifierMixin):
  """ Label Cleaner object
  
  Parameters
  ----------
  method : str or None, optional (default = removal)
    Method of cleaning the noise label.
    
  treatment_ratio : float
    Threshold for either removal or relabeling noisy labels.
    
  anomaly_detector : scikit-learn anomaly detection model or None, (default = None)
    Model for identifying anomalous instances in the dataset.
    
  base_classifier : scikit-learn classifier, (default = None)
    Classifier for predicting class.
    
  resampler : resampling method or None, (default = None)
    Sampling method for imbalance class issue.
    
  seed : int, optional ()
  
  Attributes
  ----------
  score_samples_ : array-like or None, optional (default = None)
    Anomaly score.
    
  modified_instances_ : array-like or None, optional (default = None)
    Index of the instances that are identified as likely mislabelled by AD model.
    
  removed_instances_ : array-like or None, optional (default = None)
    Index of the removed instances by the noise-label cleaning method.
    
  classes_ : ndarray of shape (n_classes, )
    List of class labels known to the classifier.
    
  Xt_, yt_: training set after treatment (if keep_treated=True)
  
  Xf_, yf_: training set after resampling (if keep_final=True)
      
  """
  def __init__(self, 
               method = None,
               treatment_ratio = 0.10,
               selftr_threshold = 0.70,
               spy_ratio = 0.10,
               anomaly_detector = None,
               high_score_anomaly = False,
               base_classifier = None, 
               resampler = None, 
               max_samples = 'auto',
               n_neighbors = 5,
               keep_treated = True, 
               keep_final = True, 
               random_state = None):
    
    self.method = method
    self.treatment_ratio = treatment_ratio
    self.spy_ratio = spy_ratio
    self.selftr_threshold = selftr_threshold
    self.anomaly_detector = anomaly_detector
    self.high_score_anomaly = high_score_anomaly
    self.base_classifier = base_classifier
    self.resampler = resampler
    self.random_state = random_state
    self.max_samples = max_samples
    self.n_neighbors = n_neighbors

    self.keep_treated = keep_treated
    self.keep_final = keep_final
      
    if (self.method not in ['selftraining', 'relabeling', 'removal',
                            'embayes','semiboost','selftraining',
                            'embayes_classifier','semiboost_classifier','selftraining_classifier', 
                            None]):
      raise ValueError('Choose an appropriate option!')

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

    self.score_samples_      = None
    self.Xt_, self.yt_       = None, None
    self.Xf_, self.yf_       = None, None
    self.modified_instances_ = None
    self.removed_instances_  = None
    self.score_samples_      = None
    self.classes_            = None
    self.ix_neg_             = None
    self.ix_neg_anm_         = None
    self.ix_rm_neg_anm_      = None
    self.ss_base_learner = None
    
    # Don't reconstruct these internal objects if we have already fitted before,
    # as we might have used the random state
    if not self._is_fitted():
      self.anomaly_detector_   = self.anomaly_detector
      self.base_classifier_    = self.base_classifier
      self.resampler_          = self.resampler
      self.random_state_       = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
      
    # De we need a default anomaly detector, base classifier or resampler?
    if self.anomaly_detector_ is None or self.anomaly_detector_ == 'iforest':
      self.anomaly_detector_ = IsolationForest(n_estimators = 100,
                                               max_samples = self.max_samples,
                                               random_state=self.random_state_, 
                                               n_jobs = -1)
    
    if self.anomaly_detector_ is None or self.anomaly_detector_ == 'lof':
      self.anomaly_detector_ = LocalOutlierFactor(n_neighbors = self.n_neighbors,
                                                  n_jobs = -1)
      
    if self.anomaly_detector_ is None or self.anomaly_detector_ == 'wiforest':
      self.anomaly_detector_ = WeightedIsoForest(n_estimators = 100,
                                                 n_neighbors = self.n_neighbors,
                                                 max_samples = self.max_samples,
                                                 random_state = self.random_state_, 
                                                 n_jobs = -1)

    if self.anomaly_detector_ is None or self.anomaly_detector_ == 'spy':
      self.anomaly_detector_ = Spy(random_state = self.random_state_)
    
  
    if self.base_classifier_ is None:
      self.base_classifier_ = RandomForestClassifier(n_estimators = 100, random_state=self.random_state_, n_jobs = -1)
    
    if self.resampler_ == 'adasyn':
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
        logging.error("This classifier only works on binary 0/1 labels, yours are {}".format(unique_y))
    
    logging.debug("Class imbalance before treatment: {}".format(np.unique(self.yt_, return_counts=True)))
    
#### STEP 1 ####

    if self.method is not None:
      try:
        if isinstance(self.anomaly_detector_, WeightedIsoForest) or isinstance(self.anomaly_detector_, Spy):
          self.anomaly_detector_.fit(self.Xt_, self.yt_)
        else:
          self.anomaly_detector_.fit(self.Xt_[self.yt_ == 0, :])
        score_samples = getattr(self.anomaly_detector_, "score_samples", None)
        if isinstance(self.anomaly_detector_, LocalOutlierFactor):
          score_samples = getattr(self.anomaly_detector_, "negative_outlier_factor_", None)
            
      except Exception as e:
        logging.error(type(self.anomaly_detector_).__name__ + ": " + repr(e))
        logging.error("Moving on without treatment, setting `score_samples` to None")
        score_samples = None
          
      if score_samples is None:
        logging.error("`anomaly_detector` does not have `score_samples` attribute or method")
      else:
        if callable(score_samples):
            self.score_samples_ = score_samples(self.Xt_[self.yt_ == 0, :])
        else:
            self.score_samples_ = score_samples
                      
        ix_neg = np.where(self.yt_ == 0)[0]
        
        if self.high_score_anomaly == True:
          adj_ratio = 1 - self.treatment_ratio
          thresh = np.quantile(self.score_samples_, adj_ratio)
          self.ix_neg_anm_ = np.where(self.score_samples_ > thresh)[0]
        
        else:
          
          adj_ratio = self.treatment_ratio
          thresh = np.quantile(self.score_samples_, adj_ratio)
          
          if isinstance(self.anomaly_detector_, Spy):
            
            # Threshold cannot be higher than the  one by Spy since it represents the higher bound
            if self.anomaly_detector_.spy_threshold_ < thresh:

              thresh = self.anomaly_detector_.spy_threshold_
              logging.warning('Threshold implied by the treatment ratio is lower than the spy threshold, the latter will be now the threshold')

          self.ix_neg_anm_ = np.where(self.score_samples_ < thresh)[0]
          
        # Seemingly Noisy Negative Instances
        self.modified_instances_ = ix_neg[self.ix_neg_anm_]

#### STEP 2 ####

    ### SIMPLE TREATMENTS

    ## Noisy Cleaning Method
    if self.method == 'removal':

      # Index of Anomalous Negative Instances
      self.removed_instances_ = self.modified_instances_
      
      # Cleaning by Removal 
      self.Xt_ = np.delete(self.Xt_, self.removed_instances_, axis = 0)
      self.yt_ = np.delete(self.yt_, self.removed_instances_, axis = 0)
    
    if self.method == 'relabeling':
      self.yt_[self.modified_instances_] = 1    
      
    if self.method == 'embayes_classifier' or  self.method == 'semiboost_classifier' or self.method == 'embayes' or self.method == 'semiboost':
      
      try:
        self.yt_[self.modified_instances_] = -1
        
      except IndexError as e:
        print(e)
        logging.warning("Anomaly Detector : " + self.anomaly_detector + " does not find unreliable instances to unlabel."  )


    ### UNLABEL-BASED TREATMENTS

    if self.method == 'tsvm':
      pass
    #   # Identify unlikely noisy negative instances as labeled input
    #   tsvm_Xt = self.Xt_
    #   tsvm_yt = self.yt_.copy()
    #   tsvm_yt[self.modified_instances_] = 0  # safeu docs states unlabeled must be marked as 0
      
    #   mask_neg_anm = np.zeros(ix_neg.shape[0], dtype=bool)
    #   mask_neg_anm[self.ix_neg_anm_] = True
    #   ix_L = ix_neg[~mask_neg_anm]
    #   ix_U = self.modified_instances_
      
    #   tsvm = TSVM(kernel='Linear')
    #   tsvm.fit(tsvm_Xt, tsvm_yt, ix_L)
    #   predicted_labels = tsvm.predict(tsvm_Xt[ix_U]).reshape(-1,)
      
    #   self.yt_[self.modified_instances_] = predicted_labels

    if self.method == 'embayes':
      
      # Identify unlikely noisy negative instances as labeled input
      emnb_Xt = self.Xt_
      emnb_yt = self.yt_.copy() 
      
      self.ss_base_learner = NaiveBayes.from_samples(NormalDistribution, emnb_Xt, emnb_yt)
      self.ss_base_learner.fit(emnb_Xt, emnb_yt)
      predicted_labels = self.ss_base_learner.predict(emnb_Xt[self.modified_instances_])
      
      self.yt_[self.modified_instances_] = predicted_labels
      
    if self.method == 'semiboost':
      
      # Identify unlikely noisy negative instances as labeled input
      smb_Xt = self.Xt_
      smb_yt = self.yt_.copy() 
      
      self.ss_base_learner = SemiBoostClassifier()
      self.ss_base_learner.fit(smb_Xt, smb_yt)
      predicted_labels = self.ss_base_learner.predict(smb_Xt[self.modified_instances_])
      
      self.yt_[self.modified_instances_] = predicted_labels

    
    elif self.method == 'selftraining':
      # According to Triguero, Garcia and Herrera (2015): max_iter = 40
      max_iter = 40 
            
      self.ss_base_learner = DecisionTreeClassifier(max_depth = 5)
      
      # Identify unlikely noisy negative instances as labeled input
      selftr_Xt_init = np.delete(self.Xt_, self.modified_instances_, axis = 0)
      selftr_yt_init = np.delete(self.yt_, self.modified_instances_, axis = 0)
      
      self.ss_base_learner.fit(selftr_Xt_init, selftr_yt_init)
      
      # Anomalous instances as unlabelled (U)
      Xt_U = self.Xt_[self.modified_instances_]
      
      probs_U = self.ss_base_learner.predict_proba(Xt_U)
      class_U = self.ss_base_learner.predict(Xt_U)
      
      # Find Least Uncertain (LU) instances given a threshold (selftr_threshold)
      probs_LUpos = probs_U[:, 1] > self.selftr_threshold
      probs_LUneg = probs_U[:, 0] > self.selftr_threshold
      
      isLU = (probs_LUpos) | (probs_LUneg)
      
      ix_LU = np.where(isLU)[0]  # index of LU instances
      
      yt_new = self.yt_.copy()
      yt_new[self.modified_instances_[ix_LU]] = class_U[ix_LU]  # Add new labels from MU instances
      
      # Create index for new instances to iterate of 
      new_modified_instances_ = list(set(self.modified_instances_) - set(self.modified_instances_[ix_LU]))
      new_modified_instances_ = np.array(new_modified_instances_)
      
      # Run while-loop of self-training algorithm
      i = 0
      while len(new_modified_instances_) != 0 and i < max_iter:
        
        Xt_Unew = np.delete(self.Xt_, new_modified_instances_, axis = 0)
        yt_Unew = np.delete(yt_new, new_modified_instances_, axis = 0)
        
        self.ss_base_learner.fit(Xt_Unew, yt_Unew)
        
        probs_Unew = self.ss_base_learner.predict_proba(self.Xt_[new_modified_instances_])
        class_Unew = self.ss_base_learner.predict(self.Xt_[new_modified_instances_])
        
        probs_Unewpos = probs_Unew[:, 0] > self.selftr_threshold
        probs_Unewneg = probs_Unew[:, 1] > self.selftr_threshold
        
        isLUnew = (probs_Unewpos) | (probs_Unewneg)
        ix_LUnew = np.where(isLUnew)[0]
        
        # If there's no more LU instances, break and remove remaining uncertain instances
        if len(ix_LUnew) == 0:
          
          self.Xt_ = np.delete(self.Xt_, new_modified_instances_, axis = 0)
          self.yt_ = np.delete(self.yt_, new_modified_instances_, axis = 0)
          
          self.removed_instances_ = new_modified_instances_
          
          break
        
        yt_new[self.modified_instances_[ix_LUnew]] = class_Unew[ix_LUnew]
        
        self.yt_ = yt_new.copy()  # Once finished the while-loop, update labels
        
        new_modified_instances_ = list(set(self.modified_instances_) - set(self.modified_instances_[ix_LUnew]))
        new_modified_instances_ = np.array(new_modified_instances_)
        
        i += 1

#### STEP 3 RESAMPLING ####
    self.Xf_, self.yf_ = self.Xt_.copy(), self.yt_.copy()
    if self.resampler_ is not None and type_of_target(self.yt_) == 'binary':
      fit_resample = getattr(self.resampler_, "fit_resample", None)
      if fit_resample is None or not callable(fit_resample):
        logging.error("Need a `fit_resample` method on your resampler {}".format(self.resampler_))
      else:
        # imblearn crashes hard if it doesn't like its options, in that case, log and continue without it
        try:
          if self.method == 'selftraining_classifier' or self.method == 'embayes_classifier' or self.method == 'semiboost_classifier':
  
            yres_ = self.yt_[self.yt_ != -1].copy()
            Xres_ = self.Xt_[self.yt_ != -1].copy()
          
            Xres_, yres_ = self.resampler_.fit_resample(Xres_, yres_)
            
            self.Xf_ = np.concatenate((Xres_, self.Xt_[self.yt_ == -1]), axis = 0)
            self.yf_ = np.concatenate((yres_, self.yt_[self.yt_ == -1]), axis = 0)   
          
          else:
             
            self.Xf_, self.yf_ = fit_resample(self.Xt_, self.yt_)
            
        except KeyboardInterrupt as e:
          raise e
          
        except Exception as e:
          logging.warning(type(self.resampler_).__name__ + ": " + repr(e))
          
    logging.debug("Class imbalance after resampling: {}".format(np.unique(self.yf_, return_counts=True)))

    #### STEP 4 MODELING ####
    if self.method == 'embayes_classifier':
      
      # Identify unlikely noisy negative instances as labeled input
      emnb_Xf = self.Xf_
      emnb_yf = self.yf_.copy() 
      
      self.ss_base_learner = NaiveBayes.from_samples(NormalDistribution, emnb_Xf, emnb_yf)
      self.ss_base_learner.fit(emnb_Xf, emnb_yf)
            
    if self.method == 'semiboost_classifier':
      
      # Identify unlikely noisy negative instances as labeled input
      smb_Xt = self.Xt_
      smb_yt = self.yt_.copy() 
      
      self.ss_base_learner = SemiBoostClassifier()
      self.ss_base_learner.fit(smb_Xt, smb_yt)
    
    if self.method == 'selftraining_classifier':
            
      self.ss_base_learner = DecisionTreeClassifier(max_depth = 5)
      
      # According to Triguero, Garcia and Herrera (2015): max_iter = 40
      max_iter = 40 
      
      unlabeled_instances = np.where(self.yf_ == -1)[0]
      
      if unlabeled_instances.size > 0: 
        # Identify unlikely noisy negative instances as labeled input
        selftr_Xf_init = np.delete(self.Xf_, unlabeled_instances, axis = 0)
        selftr_yf_init = np.delete(self.yf_, unlabeled_instances, axis = 0)
        
        self.ss_base_learner.fit(selftr_Xf_init, selftr_yf_init)
        
        # Anomalous instances as unlabelled (U)
        Xf_U = self.Xf_[unlabeled_instances]
        
        probs_U = self.ss_base_learner.predict_proba(Xf_U)
        class_U = self.ss_base_learner.predict(Xf_U)
        
        # Find Least Uncertain (LU) instances given a threshold (selftr_threshold)
        probs_LUpos = probs_U[:, 1] > self.selftr_threshold
        probs_LUneg = probs_U[:, 0] > self.selftr_threshold
        
        isLU = (probs_LUpos) | (probs_LUneg)
        
        ix_LU = np.where(isLU)[0]  # index of LU instances
        
        yf_new = self.yf_.copy()
        yf_new[unlabeled_instances[ix_LU]] = class_U[ix_LU]  # Add new labels from MU instances
        
        # Create index for new instances to iterate of 
        new_modified_instances_ = list(set(unlabeled_instances) - set(unlabeled_instances[ix_LU]))
        new_modified_instances_ = np.array(new_modified_instances_)
        
        # Run while-loop of self-training algorithm
        i = 0
        while len(new_modified_instances_) != 0 and i < max_iter:
          
          Xf_Unew = np.delete(self.Xf_, new_modified_instances_, axis = 0)
          yf_Unew = np.delete(yf_new, new_modified_instances_, axis = 0)
          
          self.ss_base_learner.fit(Xf_Unew, yf_Unew)
          
          probs_Unew = self.ss_base_learner.predict_proba(self.Xf_[new_modified_instances_])
          class_Unew = self.ss_base_learner.predict(self.Xf_[new_modified_instances_])
          
          probs_Unewpos = probs_Unew[:, 0] > self.selftr_threshold
          probs_Unewneg = probs_Unew[:, 1] > self.selftr_threshold
          
          isLUnew = (probs_Unewpos) | (probs_Unewneg)
          ix_LUnew = np.where(isLUnew)[0]
          
          # If there's no more LU instances, break and remove remaining uncertain instances
          if len(ix_LUnew) == 0:
            
            self.Xf_ = np.delete(self.Xf_, new_modified_instances_, axis = 0)
            self.yf_ = np.delete(self.yf_, new_modified_instances_, axis = 0)
            
            self.removed_instances_ = new_modified_instances_
            
            break
          
          yf_new[self.modified_instances_[ix_LUnew]] = class_Unew[ix_LUnew]
          
          self.yf_ = yf_new.copy()  # Once finished the while-loop, update labels
          
          new_modified_instances_ = list(set(self.modified_instances_) - set(self.modified_instances_[ix_LUnew]))
          new_modified_instances_ = np.array(new_modified_instances_)
          
          i += 1
      
      # If the Anomaly Detection (particularly SpyEM) provides no unreliable negatives, then just run the base model
      else:
        self.ss_base_learner.fit(self.Xf_, self.yf_)
    
    # Finally, fit the base classifier and clean up
    try:
      
      if self.method == 'selftraining_classifier' or self.method == 'embayes_classifier' or self.method == 'semiboost_classifier':
        
        self.base_classifier_ = self.ss_base_learner
      
      else: 

        self.base_classifier_.fit(self.Xf_, self.yf_)
      
      if self.method == 'embayes' or self.method == 'embayes_classifier' or self.method == 'semiboost_classifier' or self.method == 'semiboost':
        self.classes_ = ['0', '1']
        
      else:
        self.classes_ = self.base_classifier_.classes_.astype(int)
   
    except Exception as e:
      raise e
    
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
      return self.base_classifier_.predict(X)
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
      return self.base_classifier_.predict_proba(X)
    except NotFittedError as e:
      print(repr(e))
    
  def _is_fitted(self):
    return hasattr(self, 'base_classifier_')
      
  def get_params(self, deep=True):
    return {
      'method': self.method,
      'anomaly_detector': self.anomaly_detector, 
      'high_score_anomaly': self.high_score_anomaly,
      'treatment_ratio': self.treatment_ratio,
      'selftr_threshold': self.selftr_threshold,
      'resampler': self.resampler,
      'base_classifier': self.base_classifier,
      'max_samples': self.max_samples,
      'number_neighbors' : self.n_neighbors,
      'keep_treated': self.keep_treated,
      'keep_final': self.keep_final,
      'random_state': self.random_state
    }

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self