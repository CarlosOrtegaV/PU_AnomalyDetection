#from sklearn.tree.hellinger_distance_criterion import HellingerDistanceCriterion
from pos_noisyneg.utils import hyper_halfmoon
from pos_noisyneg.weighted_iforest import WeightedIsoForest

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import seaborn as sns
import pandas as pd

#### VARIABLES ####
n = 1000
f = 2
flip_ratio = 0.50
class_imbalance_ratio=0.05
feature_noise = 0.25

###############################################################################

data_2d, true_y, noisy_y, class_label = hyper_halfmoon(
                                                       npoints=n,
                                                       ndim=f,
                                                       feature_noise = feature_noise,
                                                       class_imbalance_ratio = class_imbalance_ratio,
                                                       label_noise = 'uniform',
                                                       flip_ratio = flip_ratio,
                                                       transform_axes=[0, -1], 
                                                       transform_amounts=[0.5, -.2],
                                                       random_state = 123
                                                       )

conditions = [class_label == "negs", class_label == "noisy_negs", class_label == "pos"]
choices = [ "negative", 'mislabeled negative', 'positive' ]
class_label = np.select(conditions, choices)


fig = sns.scatterplot(x = data_2d[:,0], y = data_2d[:,1], hue = class_label )
figur = fig.get_figure()    
figur.savefig('halfmoon_full.pdf', dpi=600)
sns.scatterplot(x = data_2d[:,0], y = data_2d[:,1], hue = noisy_y)

ix_tr = np.random.RandomState(123).choice(len(data_2d), size = int(0.50 * len(data_2d)), replace = False )
ix_ts = np.array(list(set(np.arange(len(data_2d))) - set(ix_tr)))
ix_neg_tr = np.intersect1d(ix_tr, np.where(noisy_y == 0)[0])
ix_pos_tr = np.intersect1d(ix_tr, np.where(noisy_y == 1)[0])
ix_neg_ts = np.intersect1d(ix_ts, np.where(noisy_y == 0)[0])
ix_pos_ts = np.intersect1d(ix_ts, np.where(noisy_y == 1)[0])
#### RECOVERING NEGATIVES ####

### Isolation Forest ###
isfo = IsolationForest(random_state = 123)
isfo.fit(data_2d[ix_neg_tr])

isfo_scores = -1*isfo.score_samples(data_2d[ix_neg_ts])

fig = plt.scatter(data_2d[ix_neg_ts,0], data_2d[ix_neg_ts,1], c = isfo_scores.reshape(-1,))

figur = fig.get_figure()    
figur.savefig('halfmoon_isofor.pdf', dpi=600)
### NNIF ###
nnif = WeightedIsoForest(random_state = 123)
nnif.fit(data_2d[ix_tr], noisy_y[ix_tr])

nnif_scores = -1*nnif.score_samples(data_2d[ix_neg_ts])

fig = plt.scatter(data_2d[ix_neg_ts,0], data_2d[ix_neg_ts,1], c = nnif_scores.reshape(-1,))

figur = fig.get_figure()    
figur.savefig('halfmoon_nnif.pdf', dpi=600)

### LOF ###

lof = LocalOutlierFactor()

lof.fit(data_2d[ix_neg_ts])

lof_scores = -1*lof.negative_outlier_factor_

fig = plt.scatter(data_2d[ix_neg_ts,0], data_2d[ix_neg_ts,1], c = lof_scores.reshape(-1,))

figur = fig.get_figure()    
figur.savefig('halfmoon_lof.pdf', dpi=600)
#### TEST ####


roc_auc_score(true_y[ix_neg_ts], isfo_scores)
average_precision_score(true_y[ix_neg_ts], isfo_scores)

roc_auc_score(true_y[ix_neg_ts], nnif_scores)
average_precision_score(true_y[ix_neg_ts], nnif_scores)

roc_auc_score(true_y[ix_neg_ts], lof_scores)
average_precision_score(true_y[ix_neg_ts], lof_scores)

