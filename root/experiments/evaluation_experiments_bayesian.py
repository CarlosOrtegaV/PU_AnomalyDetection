# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:48:41 2021

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

def bayesian_signedrank_probs(df, ref_model):
  ref = df[ref_model]
  df_ = df.drop(columns=ref_model)
  cols_ = list(df.columns)

  ix_ = cols_.index(ref_model)
  
  prob_list = []
  for col in df_.columns:
    prob = SignedRankTest.probs(ref, df_[col], rope=0.01)[0]
    prob_list.append(prob)
    
  prob_list.insert(ix_, 'ReferenceModel')
  results = dict(zip(cols_, prob_list)) 
  return results

## Load Data
df = pd.read_csv('reference_labelnoise_uniform_newNNIF.csv')

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

##
prr = ['prr_iforest_embayes', 'prr_iforest_relabeling',
       'prr_iforest_removal','prr_iforest_selftraining',
       'prr_iforest_selftraining_classifier','prr_wiforest_embayes', 
       'prr_wiforest_relabeling','prr_wiforest_removal',
       'prr_wiforest_selftraining', 'prr_wiforest_selftraining_classifier',
       'prr_lof_embayes', 'prr_lof_relabeling', 'prr_lof_removal',
       'prr_lof_selftraining','prr_lof_selftraining_classifier','prr_m1_rf', 
       'prr_m1_pubag', 'prr_m1_spyem', 'prr_m1_rnkpr', 'prr_m1_welog', 
       'prr_m1_elkno']

prc = ['prc_iforest_embayes', 'prc_iforest_relabeling',
       'prc_iforest_removal','prc_iforest_selftraining',
       'prc_iforest_selftraining_classifier','prc_wiforest_embayes', 
       'prc_wiforest_relabeling','prc_wiforest_removal',
       'prc_wiforest_selftraining', 'prc_wiforest_selftraining_classifier',
       'prc_lof_embayes', 'prc_lof_relabeling', 'prc_lof_removal',
       'prc_lof_selftraining','prc_lof_selftraining_classifier','prc_m1_rf', 
       'prc_m1_pubag', 'prc_m1_spyem', 'prc_m1_rnkpr', 'prc_m1_welog', 
       'prc_m1_elkno']

## Evaluation PR ##
df_prr_ = df_filtered[prr].rank(axis = 1, ascending = False)
df_prr_['dataset'] = df_filtered['dataset']
df_prr_ = df_filtered.groupby('dataset')[prr].mean()
prr_signedrank = bayesian_signedrank_probs(df_prr_, 'prr_wiforest_removal')

## Evaluation PU ROC-AUC ##
df_prc_ = df_filtered[prc].rank(axis = 1, ascending = False)
df_prc_['dataset'] = df_filtered['dataset']
df_prc_ = df_filtered.groupby('dataset')[prc].mean()

prc_signedrank = bayesian_signedrank_probs(df_prc_, 'prc_wiforest_relabeling')
