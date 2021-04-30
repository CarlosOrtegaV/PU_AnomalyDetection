from pos_noisyneg.ad_puforest import AdPURF
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from pos_noisyneg.ExCeeD import ExCeeD
from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.weighted_iforest import WeightedIsoForest
from pos_noisyneg.spyEM import SpyEM
from pos_noisyneg.elkannoto import WeightedElkanotoPuClassifier
from pos_noisyneg.rankpruning import RankPruning

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import ADASYN

import numpy as np
import pandas as pd
from tqdm import tqdm

#### VARIABLES ####
list_dataset = [
                pd.read_csv('scaled_cardio.csv', header = None),
                pd.read_csv('scaled_climate.csv', header = None),
                pd.read_csv('scaled_cover.csv', header = None),
                pd.read_csv('scaled_mammography.csv', header = None),
                pd.read_csv('scaled_seismic.csv', header = None),
                pd.read_csv('scaled_thyroid.csv', header = None),
                pd.read_csv('scaled_shuttle.csv', header = None),
                pd.read_csv('scaled_yeast-0-2-5-7-9_vs_3-6-8.csv', header = None),
                pd.read_csv('scaled_letter.csv', header = None),
                pd.read_csv('scaled_poker-8-9_vs_5.csv', header = None),
                pd.read_csv('scaled_winequality-red-4.csv', header = None),
                pd.read_csv('scaled_piechart2.csv', header = None),
                pd.read_csv('scaled_pizzacutter1.csv', header = None),
                pd.read_csv('scaled_satellite.csv', header = None),
                ]

list_random_state = list(np.arange(20))

list_label_noise = list(ParameterGrid({'flip_ratio': [0.25, 0.50, 0.75],
                                       'label_noise': ['knn']}))

# ROC Lists
list_roc_m0 = []
list_roc_m1_rf = []
list_roc_m1_spyem = []
list_roc_m1_rnkpr = []
list_roc_m1_pubag = []
list_roc_m1_welog = []
list_roc_m1_elkno = []

list_roc_m1_adprf_if_pb = []
list_roc_m1_adprf_nnif_pb = []

# PR Lists
list_prr_m0 = []
list_prr_m1_rf = []
list_prr_m1_spyem = []
list_prr_m1_rnkpr = []
list_prr_m1_pubag = []
list_prr_m1_welog = []
list_prr_m1_elkno = []

list_prr_m1_adprf_if_pb = []
list_prr_m1_adprf_nnif_pb = []

# PU ROC AUC Lift Lists
list_prc_m0 = []
list_prc_m1_rf = []
list_prc_m1_spyem = []
list_prc_m1_rnkpr = []
list_prc_m1_pubag = []
list_prc_m1_welog = []
list_prc_m1_elkno = []

list_prc_m1_adprf_if_pb = []
list_prc_m1_adprf_nnif_pb = []

for r in tqdm(list_random_state, desc='Data Partition'):
  for d in tqdm(list_dataset, desc='Data Set'):
    d = np.asarray(d)
    X = d[:,:-1]
    y = d[:,-1]
    rs = StratifiedShuffleSplit(n_splits = 1, test_size = 0.30, random_state = r)
    ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, y)][0]
    
    # Ideal Setting
    try:
      resampler = ADASYN(random_state=r)
      X_rs, y_rs = resampler.fit_resample(X[ix_tr], y[ix_tr])
    except:
      X_rs, y_rs = X[ix_tr], y[ix_tr]
      
    m0 = RandomForestClassifier(random_state=r)
    m0.fit(X_rs, y_rs)
    
    list_roc_m0.append(roc_auc_score(y[ix_ts], m0.predict_proba(X[ix_ts])[:,1]))
    print('ROC AUC RF No Label Noise: ', roc_auc_score(y[ix_ts],
                                                        m0.predict_proba(X[ix_ts])[:,1]) ) 
    list_prr_m0.append(average_precision_score(y[ix_ts], 
                                                m0.predict_proba(X[ix_ts])[:,1]))
    print('PR AUC RF No Label Noise: ', roc_auc_score(y[ix_ts],
                                                        m0.predict_proba(X[ix_ts])[:,1]) ) 
    

    for l in tqdm(list_label_noise, desc='Label Noise'):
      noisy_y = make_noisy_negatives(y, 
                                     X = X, 
                                     flip_ratio = l['flip_ratio'], 
                                     label_noise = l['label_noise'],
                                     n_neighbors = int(sum(y==0)),  # all true negatives as neighbors
                                     random_state = r)

      ## Class Y Generation
      class_y = y.copy()
      class_y[np.logical_and(np.array(noisy_y == 0), np.array(y == 1))] = 2
  
      conditions = [class_y == 0, class_y == 2, class_y == 1]
      values = ['negs', 'noisy_negs', 'pos']
      class_label = np.select(conditions, values)
      contamination_ratio = np.mean(class_label == 'noisy_negs')

      rs = StratifiedShuffleSplit(n_splits = 1, test_size = 0.30, 
                                  random_state = r)
      ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, class_label)][0]

      ix_neg_tr = np.intersect1d(ix_tr, np.where(noisy_y == 0)[0])
      ix_pos_tr = np.intersect1d(ix_tr, np.where(noisy_y == 1)[0])
      ix_neg_ts = np.intersect1d(ix_ts, np.where(noisy_y == 0)[0])
      ix_pos_ts = np.intersect1d(ix_ts, np.where(noisy_y == 1)[0])
      
      # Random Forest Under Label Noise
      try:
        resampler = ADASYN(random_state=r)
        X_noisy_rs, noisy_y_rs = resampler.fit_resample(X[ix_tr], noisy_y[ix_tr])
      except:
        X_noisy_rs, noisy_y_rs = X[ix_tr], noisy_y[ix_tr]
      
      m1_rf = RandomForestClassifier(random_state=r)
      m1_rf.fit(X_noisy_rs, noisy_y_rs)
      
      list_roc_m1_rf.append(roc_auc_score(y[ix_ts], 
                                          m1_rf.predict_proba(X[ix_ts])[:,1]))
      
      print('ROC AUC RF Label Noise: ', roc_auc_score(y[ix_ts], 
                                                      m1_rf.predict_proba(X[ix_ts])[:,1]) ) 

      list_prr_m1_rf.append(average_precision_score(y[ix_ts], 
                                                  m1_rf.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC RF Label Noise: ', average_precision_score(y[ix_ts], 
                                                                m1_rf.predict_proba(X[ix_ts])[:,1]) )
      
      list_prc_m1_rf.append(roc_auc_score(y[class_label != 'pos'],
                                            m1_rf.predict_proba(X[class_label != 'pos'])[:,1]))
      
      print('PU ROC RF Label Noise: ', roc_auc_score(y[class_label != 'pos'],
                                                     m1_rf.predict_proba(X[class_label != 'pos'])[:,1]))
      
      ## PU Bagging
      m1_pubag = BaggingPuClassifier(SVC(),
                                      n_estimators = 100, 
                                      n_jobs = -1, 
                                      max_samples = sum(noisy_y[ix_tr] == 1),  # Each training sample will be balanced
                                      random_state = r)
      
      m1_pubag.fit(X[ix_tr], noisy_y[ix_tr])
      list_roc_m1_pubag.append(roc_auc_score(y[ix_ts], 
                                              m1_pubag.predict_proba(X[ix_ts])[:,1]))
      
      print('ROC AUC PU Bagging: ', roc_auc_score(y[ix_ts], 
                                                  m1_pubag.predict_proba(X[ix_ts])[:,1]) ) 
      
      list_prr_m1_pubag.append(average_precision_score(y[ix_ts], 
                                                        m1_pubag.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC PU Bagging: ', average_precision_score(y[ix_ts], 
                                                            m1_pubag.predict_proba(X[ix_ts])[:,1]) ) 
      
      list_prc_m1_pubag.append(roc_auc_score(y[class_label != 'pos'],
                                              m1_pubag.predict_proba(X[class_label != 'pos'])[:,1]))
      
      print('PU ROC AUC PU Bagging: ', roc_auc_score(y[class_label != 'pos'],
                                              m1_pubag.predict_proba(X[class_label != 'pos'])[:,1]))
      
      ## SpyEM
      m1_spyem = SpyEM(random_state = r, resampler = False)
      m1_spyem.fit(X[ix_tr], noisy_y[ix_tr])
      
      list_roc_m1_spyem.append(roc_auc_score(y[ix_ts], 
                                              m1_spyem.predict_proba(X[ix_ts])[:,1]))
            
      print('ROC AUC Spy-EM: ', roc_auc_score(y[ix_ts], 
                                              m1_spyem.predict_proba(X[ix_ts])[:,1]) ) 
      list_prr_m1_spyem.append(average_precision_score(y[ix_ts], 
                                                        m1_spyem.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC Spy-EM: ', average_precision_score(y[ix_ts], 
                                                        m1_spyem.predict_proba(X[ix_ts])[:,1]) ) 
      
      list_prc_m1_spyem.append(roc_auc_score(y[class_label != 'pos'],
                                              m1_spyem.predict_proba(X[class_label != 'pos'])[:,1]))
      
      print('PU ROC AUC Spy-EM: ', roc_auc_score(y[class_label != 'pos'],
                                                  m1_spyem.predict_proba(X[class_label != 'pos'])[:,1])) 
      
      ## Weighted PU Logistic Regression
      
      m1_welog = LogisticRegression(class_weight = {0: np.mean(noisy_y[ix_tr]), 
                                                    1: 1 - np.mean(noisy_y[ix_tr]) },
                                    n_jobs = -1,
                                    penalty = 'none')
      m1_welog.fit(X[ix_tr], noisy_y[ix_tr])
      
      list_roc_m1_welog.append(roc_auc_score(y[ix_ts], 
                                              m1_welog.predict_proba(X[ix_ts])[:,1]))
            
      print('ROC AUC Weighted Log. Reg.: ', roc_auc_score(y[ix_ts], 
                                                          m1_welog.predict_proba(X[ix_ts])[:,1]) ) 
      
      list_prr_m1_welog.append(average_precision_score(y[ix_ts], 
                                                        m1_welog.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC Weighted Log. Reg.: ', average_precision_score(y[ix_ts], 
                                                                    m1_welog.predict_proba(X[ix_ts])[:,1])) 
      
      list_prc_m1_welog.append(roc_auc_score(y[class_label != 'pos'],
                                                    m1_welog.predict_proba(X[class_label != 'pos'])[:,1]))
      print('PU ROC AUC Log. Reg.: ', roc_auc_score(y[class_label != 'pos'],
                                                    m1_welog.predict_proba(X[class_label != 'pos'])[:,1])) 
      
      ## Rank Pruning
      m1_rnkpr = RankPruning(clf = LogisticRegression(penalty = 'none'), 
                              frac_neg2pos = 0)
      
      m1_rnkpr.fit(X[ix_tr], noisy_y[ix_tr])
         
      list_roc_m1_rnkpr.append(roc_auc_score(y[ix_ts], 
                                              m1_rnkpr.predict_proba(X[ix_ts])[:,1]))
            
      print('ROC AUC Rank Pruning: ', roc_auc_score(y[ix_ts], 
                                                          m1_rnkpr.predict_proba(X[ix_ts])[:,1]) ) 
      
      list_prr_m1_rnkpr.append(average_precision_score(y[ix_ts], 
                                                        m1_rnkpr.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC Rank Pruning: ', average_precision_score(y[ix_ts], 
                                                              m1_rnkpr.predict_proba(X[ix_ts])[:,1]))
      
      list_prc_m1_rnkpr.append(roc_auc_score(y[class_label != 'pos'],
                                              m1_rnkpr.predict_proba(X[class_label != 'pos'])[:,1]))
      print('PU ROC AUC Rank Pruning: ', roc_auc_score(y[class_label != 'pos'],
                                                        m1_rnkpr.predict_proba(X[class_label != 'pos'])[:,1])) 
      
      ## Elkan-Noto
      
      m1_elkno = WeightedElkanotoPuClassifier(estimator=SVC(kernel = 'linear', probability = True), 
                                              labeled= np.sum(noisy_y[ix_tr] == 1), 
                                              unlabeled = np.sum(noisy_y[ix_tr] == 0), 
                                              hold_out_ratio = 0.20)
      
      m1_elkno.fit(X[ix_tr], noisy_y[ix_tr])
      
      list_roc_m1_elkno.append(roc_auc_score(y[ix_ts], 
                                              m1_elkno.predict_proba(X[ix_ts])))
            
      print('ROC AUC Elkan-Noto: ', roc_auc_score(y[ix_ts], 
                                                    m1_elkno.predict_proba(X[ix_ts])) ) 
      
      list_prr_m1_elkno.append(average_precision_score(y[ix_ts], 
                                                        m1_elkno.predict_proba(X[ix_ts])))
      
      print('PR AUC Elkan-Noto: ', average_precision_score(y[ix_ts], 
                                                              m1_elkno.predict_proba(X[ix_ts])))
      
      list_prc_m1_elkno.append(roc_auc_score(y[class_label != 'pos'],
                                              m1_elkno.predict_proba(X[class_label != 'pos'])))
      print('PU ROC AUC Elkan-Noto: ', roc_auc_score(y[class_label != 'pos'],
                                              m1_elkno.predict_proba(X[class_label != 'pos']))) 
      
      ### Anomaly Detection Based Approach ###
      
      # Isolation Forest
      isofor = IsolationForest(random_state=r).fit(X[ix_neg_tr])
      train_score = -1* isofor.score_samples(X[ix_neg_tr])
      test_prediction = isofor.predict(X[ix_neg_tr])
      test_prediction[test_prediction == 1] = 0
      test_prediction[test_prediction == -1] = 1
      
      prob_isofor, _ = ExCeeD(train_score, train_score, test_prediction, contamination_ratio)
      
      X_tr_isofor = np.vstack((X[ix_neg_tr][prob_isofor < 0.50], X[ix_pos_tr]))
      noisy_y_tr_isofor = np.vstack((noisy_y[ix_neg_tr][prob_isofor < 0.50].reshape(-1,1), 
                                      noisy_y[ix_pos_tr].reshape(-1,1)))
      
      weight_isofor = (1 - prob_isofor[prob_isofor < 0.50])/np.sum((1 - prob_isofor[prob_isofor < 0.50]))

      # NNIF
      nnif = WeightedIsoForest(random_state = r)
      nnif.fit(X[ix_tr], noisy_y[ix_tr])
      
      train_score = -1* nnif.score_samples(X[ix_neg_tr])
      test_prediction = nnif.predict(X[ix_neg_tr])
      test_prediction[test_prediction == 1] = 0
      test_prediction[test_prediction == -1] = 1
      
      prob_nnif, _ = ExCeeD(train_score, train_score, test_prediction, contamination_ratio)

      X_tr_nnif = np.vstack((X[ix_neg_tr][prob_nnif < 0.50], X[ix_pos_tr]))
      noisy_y_tr_nnif = np.vstack((noisy_y[ix_neg_tr][prob_nnif < 0.50].reshape(-1,1), 
                                      noisy_y[ix_pos_tr].reshape(-1,1)))
      
      weight_nnif = (1 - prob_nnif[prob_nnif < 0.50])/np.sum((1 - prob_nnif[prob_nnif < 0.50]))

      # LOF
      nnif = WeightedIsoForest(random_state = r)
      nnif.fit(X[ix_tr], noisy_y[ix_tr])
      
      train_score = -1* nnif.score_samples(X[ix_neg_tr])
      test_prediction = nnif.predict(X[ix_neg_tr])
      test_prediction[test_prediction == 1] = 0
      test_prediction[test_prediction == -1] = 1
      
      prob_nnif, _ = ExCeeD(train_score, train_score, test_prediction, contamination_ratio)

      X_tr_nnif = np.vstack((X[ix_neg_tr][prob_nnif < 0.50], X[ix_pos_tr]))
      noisy_y_tr_nnif = np.vstack((noisy_y[ix_neg_tr][prob_nnif < 0.50].reshape(-1,1), 
                                      noisy_y[ix_pos_tr].reshape(-1,1)))
      
      weight_nnif = (1 - prob_nnif[prob_nnif < 0.50])/np.sum((1 - prob_nnif[prob_nnif < 0.50]))

      ## Ad PU Random Forest ##
          
      # Isolation Forest with Prob Bootstrap
      m1_adprf_if_pb = AdPURF(n_estimators = 200,
                              max_samples = np.sum(noisy_y_tr_isofor == 0),
                              random_state = r,
                              class_weight = 'balanced',
                              n_jobs = -1)
      
      m1_adprf_if_pb.fit(X_tr_isofor, noisy_y_tr_isofor.reshape(-1,), prob_score = weight_isofor)

      list_roc_m1_adprf_if_pb.append(roc_auc_score(y[ix_ts], 
                                                    m1_adprf_if_pb.predict_proba(X[ix_ts])[:,1]))
    
      print('ROC AUC ADPURF IF Prob. Bootstrap: ', roc_auc_score(y[ix_ts], 
                                                                  m1_adprf_if_pb.predict_proba(X[ix_ts])[:,1]) )   
      
      list_prr_m1_adprf_if_pb.append(average_precision_score(y[ix_ts], 
                                                            m1_adprf_if_pb.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC ADPURF IF Prob. Bootstrap: ', average_precision_score(y[ix_ts], 
                                                                          m1_adprf_if_pb.predict_proba(X[ix_ts])[:,1]))   
      
      list_prc_m1_adprf_if_pb.append(roc_auc_score(y[class_label != 'pos'],
                                                    m1_adprf_if_pb.predict_proba(X[class_label != 'pos'])[:,1]))
      print('PU ROC AUC ADPURF IF Prob. Bootstrap: ', roc_auc_score(y[class_label != 'pos'],
                                                                    m1_adprf_if_pb.predict_proba(X[class_label != 'pos'])[:,1])) 
    
      # NNIF with Prob Bootstrap
      m1_adprf_nnif_pb = AdPURF(n_estimators = 200,
                              max_samples = np.sum(noisy_y_tr_nnif == 0),
                              random_state = r,
                              class_weight = 'balanced',
                              n_jobs = -1)
      
      m1_adprf_nnif_pb.fit(X_tr_nnif, noisy_y_tr_nnif.reshape(-1,), prob_score = weight_nnif)

      list_roc_m1_adprf_nnif_pb.append(roc_auc_score(y[ix_ts], 
                                                    m1_adprf_nnif_pb.predict_proba(X[ix_ts])[:,1]))
    
      print('ROC AUC ADPURF NNIF P. Bootstrap: ', roc_auc_score(y[ix_ts], 
                                                                m1_adprf_nnif_pb.predict_proba(X[ix_ts])[:,1]) )   
      
      list_prr_m1_adprf_nnif_pb.append(average_precision_score(y[ix_ts], 
                                                                m1_adprf_nnif_pb.predict_proba(X[ix_ts])[:,1]))
      
      print('PR AUC ADPURF NNIF P. Bootstrap: ', average_precision_score(y[ix_ts], 
                                                                          m1_adprf_nnif_pb.predict_proba(X[ix_ts])[:,1]))
      
      list_prc_m1_adprf_nnif_pb.append(roc_auc_score(y[class_label != 'pos'],
                                                      m1_adprf_nnif_pb.predict_proba(X[class_label != 'pos'])[:,1]))
      print('PU ROC AUC ADPURF NNIF P. Bootstrap: ', roc_auc_score(y[class_label != 'pos'],
                                                                    m1_adprf_nnif_pb.predict_proba(X[class_label != 'pos'])[:,1]))

#### EXPORT RESULTS ####

df_experiments = pd.DataFrame()

# ROC-AUC
df_experiments['roc_m0'] = np.repeat(list_roc_m0, len(list_label_noise))
df_experiments['roc_m1_rf'] = list_roc_m1_rf
df_experiments['roc_m1_pubag'] = list_roc_m1_pubag
df_experiments['roc_m1_spyem'] = list_roc_m1_spyem
df_experiments['roc_m1_rnkpr'] = list_roc_m1_rnkpr
df_experiments['roc_m1_welog'] = list_roc_m1_welog
df_experiments['roc_m1_elkno'] = list_roc_m1_elkno

# df_experiments['roc_m1_adprf_if_pb'] = list_roc_m1_adprf_if_pb
# df_experiments['roc_m1_adprf_nnif_pb'] = list_roc_m1_adprf_nnif_pb

# PR-AUC
df_experiments['prr_m0'] = np.repeat(list_prr_m0, len(list_label_noise))
df_experiments['prr_m1_rf'] = list_prr_m1_rf
df_experiments['prr_m1_pubag'] = list_prr_m1_pubag
df_experiments['prr_m1_spyem'] = list_prr_m1_spyem
df_experiments['prr_m1_rnkpr'] = list_prr_m1_rnkpr
df_experiments['prr_m1_welog'] = list_prr_m1_welog
df_experiments['prr_m1_elkno'] = list_prr_m1_elkno

# df_experiments['prr_m1_adprf_if_pb'] = list_prr_m1_adprf_if_pb
# df_experiments['prr_m1_adprf_nnif_pb'] = list_prr_m1_adprf_nnif_pb

# PU ROC AUC
df_experiments['prc_m1_rf'] = list_prc_m1_rf
df_experiments['prc_m1_pubag'] = list_prc_m1_pubag
df_experiments['prc_m1_spyem'] = list_prc_m1_spyem
df_experiments['prc_m1_rnkpr'] = list_prc_m1_rnkpr
df_experiments['prc_m1_welog'] = list_prc_m1_welog
df_experiments['prc_m1_elkno'] = list_prc_m1_elkno

# df_experiments['prc_m1_adprf_if_pb'] = list_prc_m1_adprf_if_pb
# df_experiments['prc_m1_adprf_nnif_pb'] = list_prc_m1_adprf_nnif_pb

# Export to CSV
df_experiments.to_csv('experiment_knn_pu.csv', index=False)

# References
len_rs = len(list_random_state)
data_part_col = list(np.repeat(np.arange(len_rs), len(list_dataset)*len(list_label_noise) ))

names_datasets = [
                  'cardio', 
                  'climate', 
                  'cover', 
                  'mammography',
                  'seismic', 
                  'thyroid', 
                  'shuttle', 
                  'yeast', 
                  'letter', 
                  'poker',
                  'winequality',
                  'piechart2',
                  'pizzacutter1',
                  'satellite'
                  ]

dataset_col = list(np.repeat(names_datasets, len(list_label_noise)))*len(np.arange(len_rs))

label_noise_col = list_label_noise*len(list_random_state)*len(list_dataset)
      
df_ref = pd.DataFrame()
df_ref['data_partition'] = data_part_col
df_ref['dataset'] = dataset_col
df_ref['flip_ratio'] = [label_noise_col[i]['flip_ratio'] for i, _ in enumerate(label_noise_col)]
df_ref['label_noise'] = [label_noise_col[i]['label_noise'] for i, _ in enumerate(label_noise_col)]

df_ref.to_csv('reference_labelnoise_knn_ad.csv', index=False)