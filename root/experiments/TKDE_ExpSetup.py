# Third-Party Packages
import numpy as np
import pandas as pd

# pos_noisyneg Packgage
from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.positive_noisynegative import PNN
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from pos_noisyneg.spyEM import SpyEM
from pos_noisyneg.elkannoto import WeightedElkanotoPuClassifier
from pos_noisyneg.rankpruning import RankPruning

# scikit-learn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Third party packages
from tqdm import tqdm

#### DATA LOADING ####
df = pd.read_csv('scaled_cardio.csv', header = None)

#### DATA ####

list_seed = [120,130]  # Number of repetitions for data partition

flip_ratio = [0.25, 0.50, 0.75]

label_noise = ['uniform', 'knn']

max_ratio = 0.15
width = 0.25

# None
list_df_auc1_none = []
list_df_auc2_none = []
list_df_auc3_none = []

list_df_pr1_none = []
list_df_pr2_none = []
list_df_pr3_none = []

# Spy-EM
list_df_auc2_spyem = []
list_df_auc3_spyem = []

list_df_pr2_spyem = []
list_df_pr3_spyem = []

# PU Bagging
list_df_auc2_pubag = []
list_df_auc3_pubag = []

list_df_pr2_pubag = []
list_df_pr3_pubag = []

# PU Weighted Logistic
list_df_auc2_pulog = []
list_df_auc3_pulog = []

list_df_pr2_pulog = []
list_df_pr3_pulog = []

# Rank Pruning
list_df_auc2_rprun = []
list_df_auc3_rprun = []

list_df_pr2_rprun = []
list_df_pr3_rprun = []

# Weighted Elkan-Noto
list_df_auc2_elkno = []
list_df_auc3_elkno = []

list_df_pr2_elkno = []
list_df_pr3_elkno = []

# AD Two-Step Method
list_df_auc2 = []
list_df_auc3 = []

list_df_pr2 = []
list_df_pr3 = []

### First Preprocessing ###

poscol_label = df.shape[1] - 1 # The label column is always the last column
df = df.rename(columns = {poscol_label: "label"})

for noise in tqdm(label_noise, desc = "Kind of Noise Loop"):
  for m in tqdm(flip_ratio, desc = "Flip Ratio Loop"):
    
    j = 0
    
    #### HYPERPARAMETER CONFIGURATION #####
    
    pol_ratio = (m*np.mean(df["label"]))/(1-np.mean(df["label"])*(1-m))
    A_ = [i*pol_ratio*(1-width)/5 for i in np.arange(1, 5)]
    B_ = [pol_ratio*(1-width) + i*width*pol_ratio/6 for i in np.arange(13)]
    C_ = [pol_ratio*(1+width) + i*(max_ratio - pol_ratio*(1+width))/4 for i in np.arange(1,5)]
    
    treatment_ratio_ = np.asarray((A_ + B_ + C_), dtype = 'float32')
    
    hyperparameters_iso = {'treatment_ratio': list(treatment_ratio_),
                           'anomaly_detector': ['iforest'],
                           'method': ['selftraining', 'relabeling', 'removal',
                                      'embayes','semiboost','selftraining',
                                      'embayes_classifier','semiboost_classifier',
                                      'selftraining_classifier'],
                           'max_samples': [256],
                           'n_neighbors':[5]}
    
    list_iso = list(ParameterGrid(hyperparameters_iso)) 
    
    
    hyperparameters_nnif = {'treatment_ratio': list(treatment_ratio_),
                            'anomaly_detector': ['wiforest'],
                            'method': ['selftraining', 'relabeling', 'removal',
                                       'embayes','semiboost','selftraining',
                                       'embayes_classifier','semiboost_classifier','selftraining_classifier'],
                            'max_samples': [256],
                            'n_neighbors':[10, 20]}
    
    list_nnif = list(ParameterGrid(hyperparameters_nnif)) 
    
    hyperparameters_lof = {'treatment_ratio': list(treatment_ratio_),
                           'anomaly_detector': ['lof'],
                           'method': ['selftraining', 'relabeling', 'removal',
                                      'embayes','semiboost','selftraining',
                                      'embayes_classifier','semiboost_classifier','selftraining_classifier'],
                           'max_samples': [128],
                           'n_neighbors':[50, 100]}
    
    list_lof = list(ParameterGrid(hyperparameters_lof)) 
    
    hyperparameters = list_iso + list_nnif + list_lof
    
    
    for i in tqdm(list_seed, desc = "Data Partition Loop"):
      
      # Create empty df to drop intermediate results
      df_experiments = pd.DataFrame()
      
      pos = df[df["label"] == 1].copy()
      neg = df[df["label"] == 0].copy()
    
      # Generate Noisy Labels
      noisy_labels = make_noisy_negatives(df['label'], 
                                          X = df.drop(columns=['label']),
                                          flip_ratio = m, 
                                          label_noise = noise,
                                          random_seed = i)
                  
      pos['noisy_label'] = noisy_labels
      neg['noisy_label'] = neg["label"]
    
      # Add a variable Stratified Hold-out method
      pos['class'] = pos['noisy_label'] + 1
      neg['class'] = neg["label"]
      
      df_noisy = pd.concat([pos.reset_index(drop = True), neg.reset_index(drop = True)])
    
      df_y_noisy = df_noisy['noisy_label']
      df_X = df_noisy
    
    ### Split into Train and Test ###
    
      df_X_tr, df_X_ts, df_y_noisy_tr, df_y_noisy_ts = train_test_split(df_X, df_y_noisy, 
                                                                        test_size = 0.3,
                                                                        stratify = df_X['class'],
                                                                        random_state = np.random.RandomState(i))


      # Create Original Labels
      df_y_orgnl_tr = np.asarray(df_X_tr["label"], dtype = int)
      df_y_orgnl_ts = np.asarray(df_X_ts["label"], dtype = int)
    
      df_X_tr = df_X_tr.drop(columns = ['label','noisy_label','class'])
      df_X_ts = df_X_ts.drop(columns = ['label','noisy_label','class'])
      
      df_y_noisy_tr = np.asarray(df_y_noisy_tr)
      df_y_noisy_ts = np.asarray(df_y_noisy_ts)
      df_X_tr = np.asarray(df_X_tr)
      df_X_ts = np.asarray(df_X_ts)
          
      #### EXPERIMENTS ON CONFIGURATIONS ####
      
      # Case One : Original Training and Original Test
      # Case Two : Noisy Training and Original Test
      # Case Three : Noisy Training and Noisy Test
      
      ### INIT MODELS & TRAINING ###
    
      ## Model0 : Using Method None
      model1_none = PNN(method = None,
                        base_classifier = RandomForestClassifier(n_estimators = 100, random_state = i), 
                        resampler = 'adasyn', random_state = i)
      
      model2_none = PNN(method = None,
                        base_classifier = RandomForestClassifier(n_estimators = 100, random_state = i), 
                        resampler = 'adasyn', random_state = i)
  
      model3_none = PNN(method = None,
                        base_classifier = RandomForestClassifier(n_estimators = 100, random_state = i), 
                        resampler = 'adasyn', random_state = i)
      
      model1_none.fit(df_X_tr, df_y_orgnl_tr)
      model2_none.fit(df_X_tr, df_y_noisy_tr)
      model3_none.fit(df_X_tr, df_y_noisy_tr)
      
      ## Model SpyEM
      model2_spyem = SpyEM(resampler = False, random_state = i)      
      model3_spyem = SpyEM(resampler = False, random_state = i)
  
      model2_spyem.fit(df_X_tr, df_y_noisy_tr)
      model3_spyem.fit(df_X_tr, df_y_noisy_tr)
      
      ## Model PU Bagging (Mordelet and Vert; 2014)
      model2_pubag = BaggingPuClassifier(SVC(kernel = 'linear', random_state = i),
                                         n_estimators = 100, 
                                         n_jobs = -1, 
                                         max_samples = sum(df_y_noisy_tr == 1),  # Each training sample will be balanced
                                         random_state = i
                                         )
      
      model3_pubag = BaggingPuClassifier(SVC(kernel = 'linear', random_state = i),
                                         n_estimators = 100, 
                                         n_jobs = -1, 
                                         max_samples = sum(df_y_noisy_tr == 1),  # Each training sample will be balanced
                                         random_state = i
                                         )
      
      model2_pubag.fit(df_X_tr, df_y_noisy_tr)
      model3_pubag.fit(df_X_tr, df_y_noisy_tr)
      
      ## Model PU Weighted Logistic Regression (Lee and Liu; 2003)
      
      model2_pulog = LogisticRegression(class_weight = {0: np.mean(df_y_noisy_tr), 
                                                        1: 1 - np.mean(df_y_noisy_tr) },
                                        n_jobs = -1)
      model3_pulog = LogisticRegression(class_weight = {0: np.mean(df_y_noisy_tr), 
                                                        1: 1 - np.mean(df_y_noisy_tr) },
                                        n_jobs = -1)
      
      model2_pulog.fit(df_X_tr, df_y_noisy_tr)
      model3_pulog.fit(df_X_tr, df_y_noisy_tr)
      
      ## Model Rank Pruning (Northcutt, Wu, Chuang; 2018)
      model2_rprun = RankPruning(clf = LogisticRegression(), 
                                 frac_neg2pos = 0)
      model3_rprun = RankPruning(clf = LogisticRegression(), 
                                 frac_neg2pos = 0)

      model2_rprun.fit(df_X_tr, df_y_noisy_tr)
      model3_rprun.fit(df_X_tr, df_y_noisy_tr)
      
      ## Model Weighted Elkan-Noto  (Elkan, Noto; 2008)
      model2_elkno = WeightedElkanotoPuClassifier(
                                                  estimator=SVC(kernel = 'linear', probability = True), 
                                                  labeled= np.sum(df_y_noisy_tr == 1), 
                                                  unlabeled = np.sum(df_y_noisy_tr == 0), 
                                                  hold_out_ratio = 0.20)
      
      model3_elkno = WeightedElkanotoPuClassifier(
                                                  estimator=SVC(kernel = 'linear', probability = True), 
                                                  labeled= np.sum(df_y_noisy_tr == 1), 
                                                  unlabeled = np.sum(df_y_noisy_tr == 0), 
                                                  hold_out_ratio = 0.20)
      
      model2_elkno.fit(df_X_tr, df_y_noisy_tr)
      model3_elkno.fit(df_X_tr, df_y_noisy_tr)
  
      ### AUC & A. PRECISION ### 
      
      ## Calculate probabilities
  
      cols = [str(a) for a in model1_none.classes_]
      
      # None Models
      probs1_none = pd.DataFrame(model1_none.predict_proba(df_X_ts), columns= cols)['1']
      probs2_none = pd.DataFrame(model2_none.predict_proba(df_X_ts), columns= cols)['1']
      probs3_none = pd.DataFrame(model3_none.predict_proba(df_X_ts), columns= cols)['1']
      
      # SpyEM
      probs2_spyem = pd.DataFrame(model2_spyem.predict_proba(df_X_ts), columns= cols)['1']
      probs3_spyem = pd.DataFrame(model3_spyem.predict_proba(df_X_ts), columns= cols)['1']
      
      # PU Bagging
      probs2_pubag = pd.DataFrame(model2_pubag.predict_proba(df_X_ts), columns= cols)['1']
      probs3_pubag = pd.DataFrame(model3_pubag.predict_proba(df_X_ts), columns= cols)['1']
      
      # PU Weighted Logistic Reg
      probs2_pulog = pd.DataFrame(model2_pulog.predict_proba(df_X_ts), columns= cols)['1']
      probs3_pulog = pd.DataFrame(model3_pulog.predict_proba(df_X_ts), columns= cols)['1']
      
      # Rank Pruning
      probs2_rprun = pd.DataFrame(model2_rprun.predict_proba(df_X_ts), columns= cols)['1']
      probs3_rprun = pd.DataFrame(model3_rprun.predict_proba(df_X_ts), columns= cols)['1']
      
      # PU Weighted Elkan-Noto
      probs2_elkno = pd.DataFrame(model2_elkno.predict_proba(df_X_ts), columns= ['1'])['1']
      probs3_elkno = pd.DataFrame(model3_elkno.predict_proba(df_X_ts), columns= ['1'])['1']
      
      ## Calculate AUC & A. Precision
      
      # None Models
      auc1_none = np.around(roc_auc_score(df_y_orgnl_ts, probs1_none), decimals = 6)
      auc2_none = np.around(roc_auc_score(df_y_orgnl_ts, probs2_none), decimals = 6)
      auc3_none = np.around(roc_auc_score(df_y_noisy_ts, probs3_none), decimals = 6)
  
      list_df_auc1_none.append(auc1_none)
      list_df_auc2_none.append(auc2_none)
      list_df_auc3_none.append(auc3_none)
      
      pr1_none = np.around(average_precision_score(df_y_orgnl_ts, probs1_none), decimals = 6)
      pr2_none = np.around(average_precision_score(df_y_orgnl_ts, probs2_none), decimals = 6)
      pr3_none = np.around(average_precision_score(df_y_noisy_ts, probs3_none), decimals = 6)
  
      list_df_pr1_none.append(pr1_none)
      list_df_pr2_none.append(pr2_none)
      list_df_pr3_none.append(pr3_none)
      
      # SpyEM
      try:
        auc2_spyem = np.around(roc_auc_score(df_y_orgnl_ts, probs2_spyem), decimals = 6)
        auc3_spyem = np.around(roc_auc_score(df_y_noisy_ts, probs3_spyem), decimals = 6)
      except:
        auc2_spyem = 0
        auc3_spyem = 0
        
      list_df_auc2_spyem.append(auc2_spyem)
      list_df_auc3_spyem.append(auc3_spyem)
      
      print('Spy-EM - AUC Sc.2 : ', auc2_spyem)
      
      try:
        pr2_spyem = np.around(average_precision_score(df_y_orgnl_ts, probs2_spyem), decimals = 6)
        pr3_spyem = np.around(average_precision_score(df_y_noisy_ts, probs3_spyem), decimals = 6)
      except:
        pr2_spyem = 0
        pr3_spyem = 0
        
      list_df_pr2_spyem.append(pr2_spyem)
      list_df_pr3_spyem.append(pr3_spyem)
      
      print('Spy-EM - PR Sc.2 : ', pr2_spyem)
      
      # PU Bagging
      auc2_pubag = np.around(roc_auc_score(df_y_orgnl_ts, probs2_pubag), decimals = 6)
      auc3_pubag = np.around(roc_auc_score(df_y_noisy_ts, probs3_pubag), decimals = 6)
  
      list_df_auc2_pubag.append(auc2_pubag)
      list_df_auc3_pubag.append(auc3_pubag)
  
      print('PU Bagging - AUC Sc.2 : ', auc2_pubag)

      pr2_pubag = np.around(average_precision_score(df_y_orgnl_ts, probs2_pubag), decimals = 6)
      pr3_pubag = np.around(average_precision_score(df_y_noisy_ts, probs3_pubag), decimals = 6)
  
      list_df_pr2_pubag.append(pr2_pubag)
      list_df_pr3_pubag.append(pr3_pubag)

      print('PU Bagging - PR Sc.2 : ', pr2_pubag)

      # PU Weighted Logistic Regression
      auc2_pulog = np.around(roc_auc_score(df_y_orgnl_ts, probs2_pulog), decimals = 6)
      auc3_pulog = np.around(roc_auc_score(df_y_noisy_ts, probs3_pulog), decimals = 6)
  
      list_df_auc2_pulog.append(auc2_pulog)
      list_df_auc3_pulog.append(auc3_pulog)

      print('PU Weighted Logistic - AUC Sc.2 : ', auc2_pulog)

      pr2_pulog = np.around(average_precision_score(df_y_orgnl_ts, probs2_pulog), decimals = 6)
      pr3_pulog = np.around(average_precision_score(df_y_noisy_ts, probs3_pulog), decimals = 6)
  
      list_df_pr2_pulog.append(pr2_pulog)
      list_df_pr3_pulog.append(pr3_pulog)

      print('PU Weighted Logistic - PR Sc.2 : ', pr2_pulog)

      # Rank Pruning
      auc2_rprun = np.around(roc_auc_score(df_y_orgnl_ts, probs2_rprun), decimals = 6)
      auc3_rprun = np.around(roc_auc_score(df_y_noisy_ts, probs3_rprun), decimals = 6)
  
      list_df_auc2_rprun.append(auc2_rprun)
      list_df_auc3_rprun.append(auc3_rprun)
      
      print('Rank Pruning - AUC Sc.2 : ', auc2_rprun)

      pr2_rprun = np.around(average_precision_score(df_y_orgnl_ts, probs2_rprun), decimals = 6)
      pr3_rprun = np.around(average_precision_score(df_y_noisy_ts, probs3_rprun), decimals = 6)
  
      list_df_pr2_rprun.append(pr2_rprun)
      list_df_pr3_rprun.append(pr3_rprun)

      print('Rank Pruning - PR Sc.2 : ', pr2_rprun)

      # Weighted Elkan-Noto
      auc2_elkno = np.around(roc_auc_score(df_y_orgnl_ts, probs2_elkno), decimals = 6)
      auc3_elkno = np.around(roc_auc_score(df_y_noisy_ts, probs3_elkno), decimals = 6)
  
      list_df_auc2_elkno.append(auc2_elkno)
      list_df_auc3_elkno.append(auc3_elkno)
      
      print('Weighted Elkan-Noto - AUC Sc.2 : ', auc2_elkno)

      pr2_elkno = np.around(average_precision_score(df_y_orgnl_ts, probs2_elkno), decimals = 6)
      pr3_elkno = np.around(average_precision_score(df_y_noisy_ts, probs3_elkno), decimals = 6)
  
      list_df_pr2_elkno.append(pr2_elkno)
      list_df_pr3_elkno.append(pr3_elkno)
      
      print('Weighted Elkan-Noto - PR Sc.2 : ', pr2_elkno)
      
      k = 0
      
      for hyper in tqdm(hyperparameters, desc = "Hyperparams Loop"):
    
        print("--------------------------------------")
        print('At Data Partition {0} at Flip Ratio {3} - Starting Experiments with the hyperparameters number {1}: {2}'.format(j, k, hyper, m))
        print("--------------------------------------")
         
        model2 = PNN(method = hyper['method'], 
                     treatment_ratio = hyper['treatment_ratio'], 
                     anomaly_detector = hyper['anomaly_detector'],
                     n_neighbors = hyper['n_neighbors'],
                     max_samples = hyper['max_samples'],
                     high_score_anomaly = False, 
                     base_classifier = RandomForestClassifier(), 
                     resampler = 'adasyn', 
                     random_state = i)
        
        model3 = PNN(method = hyper['method'], 
                     treatment_ratio = hyper['treatment_ratio'], 
                     anomaly_detector = hyper['anomaly_detector'],
                     n_neighbors = hyper['n_neighbors'],
                     max_samples = hyper['max_samples'],
                     high_score_anomaly = False, 
                     base_classifier = RandomForestClassifier(), 
                     resampler = 'adasyn', 
                     random_state = i)
    
        model2.fit(df_X_tr, df_y_noisy_tr)
        model3.fit(df_X_tr, df_y_noisy_tr)        
            
        ## Add AUC scores
        cols = [str(b) for b in model2.classes_]
        
        probs2 = pd.DataFrame(model2.predict_proba(df_X_ts), columns= cols)['1']
        probs3 = pd.DataFrame(model3.predict_proba(df_X_ts), columns= cols)['1']
        
        try:
          auc2 = np.around(roc_auc_score(df_y_orgnl_ts, probs2), decimals = 6)
          auc3 = np.around(roc_auc_score(df_y_noisy_ts, probs3), decimals = 6)
        except: 
          auc2 = 0
          auc3 = 0
          
        list_df_auc2.append(auc2)
        list_df_auc3.append(auc3)
        
        print('Model 2 - AUC Sc.2 : ', auc2)
        
        try:
          pr2 = np.around(average_precision_score(df_y_orgnl_ts, probs2), decimals = 6)
          pr3 = np.around(average_precision_score(df_y_noisy_ts, probs3), decimals = 6)
        except:
          pr2 = 0
          pr3 = 0
          
        list_df_pr2.append(pr2)
        list_df_pr3.append(pr3)
        
        print('Model 2 - PR Sc.2 : ', pr2)
        
        print("--------------------------------------")
        print('Finished Iteration {0} out of {1}'.format(k, len(hyperparameters)))
        print("--------------------------------------")
        k += 1
      j += 1
      
      # Save intermediate results
      df_experiments['auc1_none'] = np.repeat(list_df_auc1_none, len(hyperparameters))
      df_experiments['auc2_none'] = np.repeat(list_df_auc2_none, len(hyperparameters))
      df_experiments['auc3_none'] = np.repeat(list_df_auc3_none, len(hyperparameters))
      df_experiments['pr1_none'] = np.repeat(list_df_pr1_none, len(hyperparameters))
      df_experiments['pr2_none'] = np.repeat(list_df_pr2_none, len(hyperparameters))
      df_experiments['pr3_none'] = np.repeat(list_df_pr3_none, len(hyperparameters))
  
      df_experiments['auc2_spyem'] = np.repeat(list_df_auc2_spyem, len(hyperparameters))
      df_experiments['auc3_spyem'] = np.repeat(list_df_auc3_spyem, len(hyperparameters))
      df_experiments['pr2_spyem'] = np.repeat(list_df_pr2_spyem, len(hyperparameters))
      df_experiments['pr3_spyem'] = np.repeat(list_df_pr3_spyem, len(hyperparameters))
      
      df_experiments['auc2_pubag'] = np.repeat(list_df_auc2_pubag, len(hyperparameters))
      df_experiments['auc3_pubag'] = np.repeat(list_df_auc3_pubag, len(hyperparameters))
      df_experiments['pr2_pubag'] = np.repeat(list_df_pr2_pubag, len(hyperparameters))
      df_experiments['pr3_pubag'] = np.repeat(list_df_pr3_pubag, len(hyperparameters))
      
      df_experiments['auc2_pulog'] = np.repeat(list_df_auc2_pulog, len(hyperparameters))
      df_experiments['auc3_pulog'] = np.repeat(list_df_auc3_pulog, len(hyperparameters))
      df_experiments['pr2_pulog'] = np.repeat(list_df_pr2_pulog, len(hyperparameters))
      df_experiments['pr3_pulog'] = np.repeat(list_df_pr3_pulog, len(hyperparameters))
      
      df_experiments['auc2_rprun'] = np.repeat(list_df_auc2_rprun, len(hyperparameters))
      df_experiments['auc3_rprun'] = np.repeat(list_df_auc3_rprun, len(hyperparameters))
      df_experiments['pr2_rprun'] = np.repeat(list_df_pr2_rprun, len(hyperparameters))
      df_experiments['pr3_rprun'] = np.repeat(list_df_pr3_rprun, len(hyperparameters))
      
      df_experiments['auc2_elkno'] = np.repeat(list_df_auc2_elkno, len(hyperparameters))
      df_experiments['auc3_elkno'] = np.repeat(list_df_auc3_elkno, len(hyperparameters))
      df_experiments['pr2_elkno'] = np.repeat(list_df_pr2_elkno, len(hyperparameters))
      df_experiments['pr3_elkno'] = np.repeat(list_df_pr3_elkno, len(hyperparameters))
      
      df_experiments['auc2'] = list_df_auc2
      df_experiments['auc3'] = list_df_auc3
      df_experiments['pr2'] = list_df_pr2
      df_experiments['pr3'] = list_df_pr3
      
      df_experiments.to_csv('experiments.csv')

# Build DataFrame for Results CSV
#datasets_col = np.repeat(np.arange(len(df_list)), len(flip_ratio)*len(list_seed)*len(hyperparameters) )
labelnoise_col = list(np.repeat(label_noise, len(flip_ratio)*len(list_seed)*len(hyperparameters)*len(list_seed) ))

flip_ratio_col = list(np.repeat(flip_ratio, len(list_seed)*len(hyperparameters) ))*len(label_noise)

data_part_col = list(np.repeat(np.arange(len(list_seed)), len(hyperparameters) ))*len(label_noise)*len(flip_ratio)

hyper_col = hyperparameters*len(flip_ratio)*len(list_seed)*len(label_noise)

df_ref = pd.DataFrame()
#df_ref['dataset'] = datasets_col
df_ref['labelnoise'] = labelnoise_col

df_ref['flip'] = flip_ratio_col
df_ref['data_partition'] = data_part_col
df_ref['anomaly_detector'] = [hyper_col[i]['anomaly_detector'] for i, _ in enumerate(hyper_col)]
df_ref['method'] = [hyper_col[i]['method'] for i, _ in enumerate(hyper_col)]
df_ref['max_samples'] = [hyper_col[i]['max_samples'] for i, _ in enumerate(hyper_col)]
df_ref['n_neighbors'] = [hyper_col[i]['n_neighbors'] for i, _ in enumerate(hyper_col)]
df_ref['treatment_ratio'] = [hyper_col[i]['treatment_ratio'] for i, _ in enumerate(hyper_col)]

df_ref.to_csv('reference.csv')