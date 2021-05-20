# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:28:06 2020

@author: orteg
"""
import math
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import beta
from sklearn.preprocessing import quantile_transform
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import column_or_1d


def ave_savings_score(y_true, score, cost_mat):
    #TODO: update description
    """Savings score.
   
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    Returns
    -------
    score : float
        Savings of a using y_pred on y_true with cost-matrix cost-mat
        The best performance is 1.
    References
    ----------
    .. [1] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.
    See also
    --------
    cost_loss
    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import savings_score, cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> savings_score(y_true, y_pred, cost_mat)
    0.5
    """

    #TODO: Check consistency of cost_mat
    y_true = column_or_1d(y_true)
    score = column_or_1d(score)
    # n_samples = len(y_true)
    amount = cost_mat[:,1]
    cf = cost_mat[:,0]
    ave_savings = np.sum(y_true*score*amount - score*cf)/np.sum(y_true*amount)
  
    return ave_savings


def make_noisy_negatives(y, 
                         X = None, 
                         flip_ratio = None, 
                         label_noise = 'uniform',
                         n_neighbors = 'auto',
                         true_prop_pos = None, 
                         pollution_ratio = None,
                         random_state = None):
  
  """ Flipping Negatives into Positives
  
    Generating noisy negatives by flipping. It can generate noisy negatives 
    using a flip ratio (percentage of flipped positives) or a pollution ratio 
    (percentage of hidden positives in negatives)
    
    Parameters
    ----------
    y                 : array-like, compulsory
    X                 : array-like, sparse matrix of shape (n_samples, n_features)
    flip_ratio        : float, optional (default = None)
        Percentage of positives into noisy negatives.
    true_prop_pos     : float, optional (default = None)
        Percentage of true positives in the dataset.
    pollution_ratio   : float, optional (default = None)
        Percentage of hidden positives (noisy negs) among negatives.
    random_seed       : int, optional (default = 123)
      
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    """
  if X is not None:  
    X = check_array(X, accept_sparse='csr', dtype=np.float64)
  
  if X is None and label_noise == 'knn':
    raise ValueError("Label noise by Nearest Neighbors requires features matrix ``X`` ")
  
  random_state = check_random_state(random_state)
  
  y_ = y.copy()
  y_ = check_array(y_, ensure_2d=False, dtype=None)
  check_consistent_length(X, y_)
  n_pos = np.sum(y_ == 1)
  
  true_prop_pos = np.mean(y_)
  
  if n_neighbors == 'auto':
    n_neighbors = math.floor(math.sqrt(len(y_)))
    
  if flip_ratio is None and pollution_ratio is None:
    raise ValueError("Either flip rate (`flip_ratio`) or pollution ratio (`pollution_ratio`) must be known")
 
  if flip_ratio is not None and true_prop_pos is not None:
    flip_ratio_ = flip_ratio
       
  else:
    raise ValueError("If flip ratio is not None, true proportion of positives must be known")

  ## Types of Label Noise
  
  # Uniform Label Noise
  if label_noise == 'uniform':
    p_ = None

  # KNN Label Noise
  elif label_noise == 'knn':
    

    dist_matrix = pairwise_distances(X, n_jobs = -1)
    ix_pos = np.where(y == 1)[0]
    ix_neg = np.where(y == 0)[0]

    pos_dist_mt = dist_matrix[ix_pos][:,ix_neg]
    
    pos_dist_mt.sort(axis = 1)
    k_mean_dist = np.mean(pos_dist_mt[:,:n_neighbors], axis = 1)
    
    sample_weight = np.array([k_mean_dist[i]/np.sum(k_mean_dist) for i,_ in enumerate(k_mean_dist)])
    p_ = sample_weight.reshape(-1,)
    
    
  else: 
    raise ValueError('labeling is not valid; Use knn or uniform instead')
    
  ## Size of Flipped Negatives
  if label_noise == 'knn':
    size_ = min(int(np.ceil(flip_ratio_*n_pos)), len(np.nonzero(p_)[0]))  
    
  else: 
    size_ = int(np.ceil(flip_ratio_*n_pos))
  
  ## Index to be Flipped Positives
  indices = random_state.choice(range(n_pos), 
                                size_, 
                                replace = False, 
                                p = p_)
    
  print('Number Samples First Stage ', len(indices))

  if len(indices) < int(np.ceil(flip_ratio_*n_pos)):

      print('Sampling Second Stage - Random Sampling on Complement Index Set')

      indices_comp = np.setdiff1d(range(n_pos), indices)
      size_extra_ = int(np.ceil(flip_ratio_*n_pos)) - len(indices)
      indices_extra = random_state.choice(indices_comp,
                                          size_extra_,
                                          replace=False)
      
      print('Number Samples Second Stage ', len(indices_extra))

      indices = np.hstack((indices, indices_extra))
    
  
  pos_y = y_[y_ == 1].copy()
  pos_y[indices] = 0
  
  y_[y_ == 1] = pos_y

  mislabeled_rate = 1 - np.mean(pos_y)
  pol_ratio = (mislabeled_rate*true_prop_pos)/(1 - true_prop_pos*(1 - mislabeled_rate))

  print("Flipping ratio (pos. to neg.): {:.3f}".format(mislabeled_rate))
  print("Pollution ratio (noisy neg. among observed neg.): {:.3f}".format(pol_ratio))
  
  return y_

# Create Half-moon dataset
def hyper_halfmoon(npoints=1000, 
				   ndim=2,
                   ndim_useless = 3,
				   class_imbalance_ratio=0.20,
                   feature_noise = 0.20,
                   label_noise = 'uniform',
                   n_neighbors = 'auto',
                   flip_ratio = 0.60,
                   flip_axis=-1, 
                   transform_axes=[0], 
                   transform_amounts=[.5],
                   random_state = None):
    
  random_state = check_random_state(random_state)
  def sample_spherical(npoints, ndim, random_state):
    vec =  random_state.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return np.swapaxes(vec, 0, 1)
    
  upper = sample_spherical(int(np.ceil(npoints*(1-class_imbalance_ratio))), ndim, random_state)
  lower = sample_spherical(int(np.ceil(npoints*class_imbalance_ratio)), ndim, random_state)
  upper[:, flip_axis] =  np.abs(upper[:, flip_axis])
  lower[:, flip_axis] = -np.abs(lower[:, flip_axis])
  upper[:, transform_axes] += transform_amounts
  lower[:, transform_axes] -= transform_amounts
  
  X = np.vstack((upper, lower))
         
  true_y = np.hstack((
      np.full((len(upper)), 0),
      np.full((len(lower)), 1)
  ))

  noisy_y = make_noisy_negatives(true_y, 
                                 X=X, 
                                 flip_ratio=flip_ratio,
                                 label_noise=label_noise, 
                                 n_neighbors=n_neighbors, 
                                 random_state = random_state)
  
  ## Class Y Generation
  class_y = true_y.copy()
  class_y[np.logical_and(np.array(noisy_y == 0), np.array(true_y == 1))] = 2
  
  conditions = [class_y == 0, class_y == 2, class_y == 1]
  values = ['negs', 'noisy_negs', 'pos']
  class_labels = np.select(conditions, values)
  
  # Fill useless features
  if ndim_useless > 0:
    #random_state = np.random.RandomState(random_state)
    X = np.hstack((
                   X, 
                   random_state.randn(len(X), ndim_useless)
                  ))
        
  if feature_noise is not None:
    #random_state = np.random.RandomState(random_state)
    X += random_state.normal(scale=feature_noise, size=X.shape)
    
  return X, true_y, noisy_y, class_labels