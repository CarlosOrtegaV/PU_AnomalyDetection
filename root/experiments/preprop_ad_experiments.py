# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:17:46 2020

@author: orteg
"""
import pandas as pd
import numpy as np

### Functions ####

## Load Data
df = pd.read_csv('reference_ad_knn_newNNIF.csv')

df['technique'] = df['anomaly_detector']+'_'+df['method']
df_ = df[['dataset', 'flip_ratio', 'data_partition','technique','auc2',
          'pr2', 'pur2']]

ref_df = df_[df_['technique']=='wiforest_relabeling'][['dataset','data_partition','flip_ratio']].reset_index(drop=True)
list_aux_df = []
for i in np.unique(df_['technique']):
  
  aux_df = df_[df_['technique']==i]
  aux_df = aux_df[['auc2','pr2','pur2']]
  aux_df = aux_df.rename(columns={'auc2':'roc_'+i,
                                  'pr2':'prr_'+i,
                                  'pur2':'prc_'+i}).reset_index(drop=True)
  list_aux_df.append(aux_df)


aux_dfs = pd.concat(list_aux_df, axis = 1).reset_index(drop=True)
ref_df = pd.concat([ref_df,aux_dfs], axis = 1).reset_index(drop=True)

ref_df.to_csv('ref_ad_knn_newNNIF_models.csv',index = None)
