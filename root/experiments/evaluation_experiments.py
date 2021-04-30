# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:46 2020

@author: orteg
"""
import scipy.stats as stats
import pandas as pd
import numpy as np
from baycomp import SignedRankTest
### Functions ####

def imandavenport_test(num_models, num_datasets, model_ranks):

  chisqF = (12*num_datasets/(num_models*(num_models+1)))*(sum(model_ranks**2)-(num_models*(num_models+1)**2)/4)
  Ff = (num_datasets-1)*chisqF/(num_datasets*(num_models-1)-chisqF)

  df1 = num_models - 1
  df2 = (num_models-1)*(num_datasets-1)
  pvalue = 1 - stats.f.cdf(Ff, df1, df2)
  print('p-value ROC ranks: ', pvalue)
  return pvalue
    
def bonfholm_test(num_models, num_datasets, model_names, model_ranks, alpha):
  denominator_ = (num_models*(num_models+1)/(6*num_datasets))**0.5
  list_pv_ = np.array(model_ranks)
  ix_min_ = np.where(list_pv_==np.min(model_ranks))[0]
  list_pv = np.delete(list_pv_, ix_min_)
  model_names_ = np.delete(model_names, ix_min_)

  z_scores = np.asarray([ (list_pv[i] - np.min(model_ranks))/denominator_ for i, _ in enumerate(list_pv) ])
  p_values = stats.norm.sf(abs(z_scores)) #one-sided
  ix_sort = np.argsort(p_values)
  decision = np.ones(num_models - 1, dtype=bool)
  adj_p_values = p_values.copy()
  for m, i in enumerate(p_values[ix_sort]):  
    if i <= alpha/(num_models-1-m):
      decision[ix_sort[m]] = False
      
    adj_p_values[ix_sort[m]] *= (num_models-1-m) 
      
  return model_names[ix_min_], model_names_[decision], adj_p_values, p_values

## Load Data
df = pd.read_csv('reference_labelnoise_uniform.csv')

cols = ['data_partition','flip_ratio','label_noise','roc_m0','prr_m0']

cond = 'flip_ratio == "0.25"'
df_ = df.copy()
df_ = df_.query(cond)
df_filtered = df_.drop(columns = cols)

# Average PU ROC-AUC for Standard RF under No Label Noise
print('ROC-AUC Under No Label Noise: ', df_['roc_m0'].mean())

# Average PR-AUC for Standard RF under No Label Noise
print('PR-AUC Under No Label Noise: ', df['prr_m0'].mean())

# Average Metrics
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).to_string())

#### AD PU TECHNIQUES ####

## AD PU Techniques - Isolation Forest ##

roc_cols1_isofor = ['roc_iforest_embayes', 'roc_iforest_relabeling',
                    'roc_iforest_removal','roc_iforest_selftraining',
                    'roc_iforest_selftraining_classifier']

prr_cols1_isofor = ['prr_iforest_embayes', 'prr_iforest_relabeling',
                    'prr_iforest_removal','prr_iforest_selftraining',
                    'prr_iforest_selftraining_classifier']

prc_cols1_isofor = ['prc_iforest_embayes', 'prc_iforest_relabeling',
                    'prc_iforest_removal','prc_iforest_selftraining',
                    'prc_iforest_selftraining_classifier']

## AD PU Techniques - NNIF ##

roc_cols1_nnif = ['roc_wiforest_embayes', 'roc_wiforest_relabeling',
                    'roc_wiforest_removal','roc_wiforest_selftraining',
                    'roc_wiforest_selftraining_classifier']

prr_cols1_nnif = ['prr_wiforest_embayes', 'prr_wiforest_relabeling',
                    'prr_wiforest_removal','prr_wiforest_selftraining',
                    'prr_wiforest_selftraining_classifier']

prc_cols1_nnif = ['prc_wiforest_embayes', 'prc_wiforest_relabeling',
                    'prc_wiforest_removal','prc_wiforest_selftraining',
                    'prc_wiforest_selftraining_classifier']

## AD PU Techniques - LOF ##

roc_cols1_lof = ['roc_lof_embayes', 'roc_lof_relabeling',
                 'roc_lof_removal','roc_lof_selftraining',
                 'roc_lof_selftraining_classifier']

prr_cols1_lof = ['prr_lof_embayes', 'prr_lof_relabeling',
                    'prr_lof_removal','prr_lof_selftraining',
                    'prr_lof_selftraining_classifier']

prc_cols1_lof = ['prc_lof_embayes', 'prc_lof_relabeling',
                 'prc_lof_removal','prc_lof_selftraining',
                 'prc_lof_selftraining_classifier']

## Evaluation AD-based models ##

df_prr_ = df_filtered[prr_cols1_lof].copy().rank(axis = 1, ascending = False)
df_prr_['dataset'] = df_['dataset']
df_prr_ = df_prr_.groupby('dataset').mean()

df_prc_ = df_filtered[prc_cols1_lof].copy().rank(axis = 1, ascending = False)
df_prc_['dataset'] = df_['dataset']
df_prc_ = df_prc_.groupby('dataset').mean()

df_prr_transpose = df_prr_.T
df_prc_transpose = df_prc_.T

df_prr_rank_mean_ = df_prr_transpose.mean(axis = 1)
df_prc_rank_mean_ = df_prc_transpose.mean(axis = 1)

# Variables for Multiple Comparison Test
N = len(df_prr_.index)
k = len(df_prr_.columns)

## PR-AUC
model_names_ = np.asarray(list(df_prr_rank_mean_.index))

imandavenport_test(k,N, df_prr_rank_mean_)

# Holm's Test PR
lowest_rank_pr, no_rejected_pr, adjpvalues_pr, pvalues_pr = bonfholm_test(k, N, model_names_, df_prr_rank_mean_, 0.05)

## PU ROC 
model_names_ = np.asarray(list(df_prc_rank_mean_.index))

imandavenport_test(k,N, df_prc_rank_mean_)

# Holm's Test PU ROC
lowest_rank_prc, no_rejected_prc, adjpvalues_prc, pvalues_prc = bonfholm_test(k, N, model_names_, df_prc_rank_mean_, 0.05)

#### PU LEARNING TECHNIQUES ####

prr_cols = ['prr_m1_rf', 'prr_m1_pubag', 'prr_m1_spyem',
           'prr_m1_rnkpr', 'prr_m1_welog', 'prr_m1_elkno']

prc_cols = ['prc_m1_rf', 'prc_m1_pubag', 'prc_m1_spyem',
           'prc_m1_rnkpr', 'prc_m1_welog', 'prc_m1_elkno']

df_prr_ = df_filtered[prr_cols].copy().rank(axis = 1, ascending = False)
df_prr_['dataset'] = df_['dataset']
df_prr_ = df_prr_.groupby('dataset').mean()

df_prc_ = df_filtered[prc_cols].copy().rank(axis = 1, ascending = False)
df_prc_['dataset'] = df_['dataset']
df_prc_ = df_prc_.groupby('dataset').mean()

df_prr_transpose = df_prr_.T
df_prc_transpose = df_prc_.T

df_prr_rank_mean_ = df_prr_transpose.mean(axis = 1)
df_prc_rank_mean_ = df_prc_transpose.mean(axis = 1)

## Variables for Multiple Comparison Test
N = len(df_prc_.index)
k = len(df_prc_.columns)

## PR-AUC 
model_prr_names_ = np.asarray(list(df_prr_rank_mean_.index))

# Iman Davenport Test
imandavenport_test(k,N, df_prr_rank_mean_)

# Holm's Test PR
lowest_rank_pr, no_rejected_pr, adjpvalues_pr, pvalues_pr = bonfholm_test(k, 
                                                                          N, 
                                                                          model_prr_names_, 
                                                                          df_prr_rank_mean_, 
                                                                          0.10)

## PU ROC 
model_prc_names_ = np.asarray(list(df_prc_rank_mean_.index))

imandavenport_test(k,N, df_prc_rank_mean_)

# Holm's Test PU ROC
lowest_rank_prc, no_rejected_prc, adjpvalues_prc, pvalues_prc = bonfholm_test(k,
                                                                              N, 
                                                                              model_prc_names_, 
                                                                              df_prc_rank_mean_, 
                                                                              0.10)

