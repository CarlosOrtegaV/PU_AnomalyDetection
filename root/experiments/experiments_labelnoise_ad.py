# Third-Party Packages
import numpy as np
import pandas as pd

# pos_noisyneg Packgage
from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.positive_noisynegative import PNN

# scikit-learn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier

# Third party packages
from tqdm import tqdm

#### DATA LOADING ####

#### DATA ####

list_random_state = list(np.arange(20))  # Number of repetitions for data partition

flip_ratio = [0.25, 0.50, 0.75]

label_noise = ['uniform']
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
# AD Two-Step Method
list_df_auc2 = []

list_df_pr2 = []

list_df_pur2 = []

### First Preprocessing ###
for df in tqdm(list_dataset, desc = 'Dataset Loop'):
  
  poscol_label = df.shape[1] - 1 # The label column is always the last column
  df = df.rename(columns = {poscol_label: "label"})
  
  for noise in tqdm(label_noise, desc = "Kind of Noise Loop"):
    for m in tqdm(flip_ratio, desc = "Flip Ratio Loop"):
      
      j = 0
      
      #### HYPERPARAMETER CONFIGURATION #####
      
      pol_ratio = (m*np.mean(df["label"]))/(1-np.mean(df["label"])*(1-m))
          
      # hyperparameters_iso = {'treatment_ratio': [pol_ratio],
      #                        'anomaly_detector': ['iforest'],
      #                        'method': ['selftraining', 'relabeling', 'removal',
      #                                   'embayes','selftraining_classifier'],
      #                        'max_samples': [256],
      #                        'n_neighbors':[5]}
      
      # list_iso = list(ParameterGrid(hyperparameters_iso)) 
      
      
      hyperparameters_nnif = {'treatment_ratio': [pol_ratio],
                              'anomaly_detector': ['wiforest'],
                              'method': ['selftraining', 'relabeling', 'removal',
                                         'embayes','selftraining_classifier'],
                              'max_samples': [256],
                              'n_neighbors':[5]}
      
      list_nnif = list(ParameterGrid(hyperparameters_nnif)) 
      
      # hyperparameters_lof = {'treatment_ratio': [pol_ratio],
      #                        'anomaly_detector': ['lof'],
      #                        'method': ['selftraining', 'relabeling', 'removal',
      #                                   'embayes','selftraining_classifier'],
      #                        'max_samples': [256],
      #                        'n_neighbors':[20]}
      
      # list_lof = list(ParameterGrid(hyperparameters_lof)) 
      
      # hyperparameters = list_iso + list_nnif + list_lof
      hyperparameters = list_nnif
      
      for i in tqdm(list_random_state, desc = "Data Partition Loop"):
        noisy_y = make_noisy_negatives(df['label'], 
                                       X = df.drop(columns=['label']), 
                                       flip_ratio = m, 
                                       label_noise = noise,
                                       n_neighbors = int(np.sum(df['label']==0)),  # all true negatives as neighbors
                                       random_state = i)
  
        ## Class Y Generation
        class_y = df['label'].copy()
        class_y[np.logical_and(np.array(noisy_y == 0), np.array(df['label'] == 1))] = 2
    
        conditions = [class_y == 0, class_y == 2, class_y == 1]
        values = ['negs', 'noisy_negs', 'pos']
        class_label = np.select(conditions, values)
                    
      ### Split into Train and Test ###
      
        df_X_tr, df_X_ts, df_y_noisy_tr, df_y_noisy_ts = train_test_split(df, noisy_y, 
                                                                          test_size = 0.3,
                                                                          stratify = class_label,
                                                                          random_state = np.random.RandomState(i))
  
  
        # Create Original Labels
        
        df_X = df.drop(columns = ['label'])
        
        df_y_orgnl_tr = np.asarray(df_X_tr["label"], dtype = int)
        df_y_orgnl_ts = np.asarray(df_X_ts["label"], dtype = int)
      
        df_X_tr = df_X_tr.drop(columns = ['label'])
        df_X_ts = df_X_ts.drop(columns = ['label'])
        
        df_y_noisy_tr = np.asarray(df_y_noisy_tr)
        df_y_noisy_ts = np.asarray(df_y_noisy_ts)
        df_X_tr = np.asarray(df_X_tr)
        df_X_ts = np.asarray(df_X_ts)
            
        #### EXPERIMENTS ON CONFIGURATIONS ####
        
        # Case One : Original Training and Original Test
        # Case Two : Noisy Training and Original Test
        # Case Three : Noisy Training and Noisy Test
        
        ### INIT MODELS & TRAINING ###
        
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
                       base_classifier = RandomForestClassifier(random_state = i), 
                       resampler = 'adasyn', 
                       random_state = i)
        
      
          model2.fit(df_X_tr, df_y_noisy_tr)
              
          ## Add AUC scores ##
          cols = [str(b) for b in model2.classes_]
          
          probs2 = pd.DataFrame(model2.predict_proba(df_X_ts), columns= cols)['1']
          
          try:
            auc2 = np.around(roc_auc_score(df_y_orgnl_ts, probs2), decimals = 6)
          except: 
            auc2 = 0            
          list_df_auc2.append(auc2)
          
          print('AUC Sc.2 : ', auc2)
          
          ## Add PR scores ##
          try:
            pr2 = np.around(average_precision_score(df_y_orgnl_ts, probs2), decimals = 6)
          except:
            pr2 = 0
            
          list_df_pr2.append(pr2)
          
          print('PR Sc.2 : ', pr2)
          
          ## Add PU ROC scores ##
          try:
            pur2 = np.around(roc_auc_score(df['label'][class_label != 'pos'], 
                                           model2.predict_proba(df_X[class_label != 'pos'])[:,1]),
                             decimals = 6)
            
          except:
            pur2 = 0
        
          list_df_pur2.append(pur2)
          print('PU ROC Sc.2 : ', pur2)

          print("--------------------------------------")
          print('Finished Iteration {0} out of {1}'.format(k, len(hyperparameters)))
          print("--------------------------------------")
          k += 1
        j += 1
        
# Save results
# Create empty df to drop intermediate results
df_experiments = pd.DataFrame()
df_experiments['auc2'] = list_df_auc2
df_experiments['pr2'] = list_df_pr2
df_experiments['pur2'] = list_df_pur2

df_experiments.to_csv('ad_experiments_knn_newNNIF.csv', index=False)

# Build DataFrame for Results CSV

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
                 

datasets_col = np.repeat(names_datasets, len(flip_ratio)*len(list_random_state)*len(hyperparameters) )
labelnoise_col = list(np.repeat(label_noise, len(flip_ratio)*len(list_random_state)*len(hyperparameters)))*len(names_datasets)

flip_ratio_col = list(np.repeat(flip_ratio, len(list_random_state)*len(hyperparameters) ))*len(label_noise)*len(names_datasets)

data_part_col = list(np.repeat(np.arange(len(list_random_state)), len(hyperparameters) ))*len(label_noise)*len(flip_ratio)*len(names_datasets)

hyper_col = hyperparameters*len(flip_ratio)*len(list_random_state)*len(label_noise)*len(names_datasets)

df_ref = pd.DataFrame()
df_ref['dataset'] = datasets_col
df_ref['labelnoise'] = labelnoise_col

df_ref['flip_ratio'] = flip_ratio_col
df_ref['data_partition'] = data_part_col
df_ref['anomaly_detector'] = [hyper_col[i]['anomaly_detector'] for i, _ in enumerate(hyper_col)]
df_ref['method'] = [hyper_col[i]['method'] for i, _ in enumerate(hyper_col)]
df_ref['max_samples'] = [hyper_col[i]['max_samples'] for i, _ in enumerate(hyper_col)]
df_ref['n_neighbors'] = [hyper_col[i]['n_neighbors'] for i, _ in enumerate(hyper_col)]
df_ref['treatment_ratio'] = [hyper_col[i]['treatment_ratio'] for i, _ in enumerate(hyper_col)]

df_ref.to_csv('reference_ad_knn_newNNIF.csv', index=False)