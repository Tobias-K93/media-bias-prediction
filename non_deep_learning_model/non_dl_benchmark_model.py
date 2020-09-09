########### Non-DL Benchmark model ##########
import time
import multiprocessing
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from joblib import dump, load
############################
### choosing split ratio ###
split_ratio = (80,10,10) ###
######################################
affix = 'allsides_duplicates_removed'#
######################################

non_dl_directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(non_dl_directory_path)

### loading target variable
y = np.load(f'non_dl_{affix}_bias.npy', allow_pickle=True)

# changing string labels to ints
allsides_bias_dict = {'Left': 0, 'Lean Left': 1, 'Center': 2, 'Lean Right': 3, 'Right':4}

for label in allsides_bias_dict:
    y[y==label] = allsides_bias_dict[label]
y = y.astype('int')

### loading linguistic variables
character_variables = np.load(f'non_dl_{affix}_character_variables.npy')
dict_count_variables = np.load(f'non_dl_{affix}_dict_count_variables.npy')
token_based_variables = np.load(f'non_dl_{affix}_token_based_variables.npy')
pos_variables = np.load(f'non_dl_{affix}_pos_variables.npy')

X = np.concatenate((character_variables,dict_count_variables,
                    token_based_variables, pos_variables),axis=1)

# replacing single Nan value with 0 (Honor's R could not be calculated in one case)
X = np.nan_to_num(X)

### removing unwanted sources ##################################################
sources = np.load(f'non_dl_{affix}_source.npy', allow_pickle=True)

wrongly_labeled = ['RightWingWatch']
news_aggregators = ['Drudge Report', 'Real Clear Politics', 'Yahoo News'] 
tabloids = ['New York Daily News', 'Daily Mail', 'New York Post']
unwanted_sources = wrongly_labeled + tabloids + news_aggregators

boolean_array = np.full((len(sources), ), False)

for source in unwanted_sources:
    boolean_array += sources==source

# boolean to remove unwanted sources
inverted_boolean_array = np.invert(boolean_array)

y = y[inverted_boolean_array]
X = X[inverted_boolean_array]
sources = sources[inverted_boolean_array]
################################################################################

### creating id vectors
np.random.seed(123)
data_length = len(y) 

ids = np.arange(data_length)
np.random.shuffle(ids) 
# cut offs for train- validation- and test-set according to split ratio
train_val_cut = int(data_length*(split_ratio[0]/100))
val_test_cut = int(data_length*(split_ratio[0]+split_ratio[1])/100)
# ids for each set
train_ids = ids[:train_val_cut]
val_ids = ids[train_val_cut:val_test_cut]
test_ids = ids[val_test_cut:]

### Creating train- val- test-sets
X_train = X[train_ids]
X_val = X[val_ids]
X_test = X[test_ids]

y_train = y[train_ids]
y_val = y[val_ids]
y_test = y[test_ids]

### Setting up and training random forest
cpus = multiprocessing.cpu_count()

### hyper parameter search
# parameters = {'n_estimators': [300, 500],
#               'max_depth': [None,20,30,40,50,70],
#               'min_samples_split': [2,6,10,20],
#               'max_features': ['sqrt',27,None],
#               'max_samples': [0.2,0.5,0.7,None]}

# param_grid = ParameterGrid(parameters)

# grid_search_iterations = 10

# random_grid_ids = np.random.randint(0,len(param_grid)-1,grid_search_iterations)

# random_grid_scores = []
# for i,id in enumerate(random_grid_ids):
#     random_params = param_grid[id]

#     rf = RandomForestClassifier(n_estimators=random_params['n_estimators'],
#                                 min_samples_split=random_params['min_samples_split'],
#                                 max_features= random_params['max_features'],
#                                 max_depth= random_params['max_depth'],
#                                 max_samples=random_params['max_samples'],
#                                 n_jobs=max(1,cpus-1), 
#                                 verbose=0)

#     print(pd.DataFrame(random_params, index=[i+1]))
#     print('------------------------------'*2)
#     start = time.time()
#     rf.fit(X_train, y_train)
#     end = time.time()-start
#     minutes = end // 60
#     seconds = end % 60
#     print('Time: %i minutes %i seconds'%(minutes, seconds))

#     # predict on train 
#     train_predictions =  rf.predict(X_train)
#     train_acc = np.sum(y_train==train_predictions)/len(y_train)
#     train_f1 = f1_score(y_train, train_predictions, average='macro')

#     # predict on validation
#     val_predictions = rf.predict(X_val)
#     val_acc = np.sum(y_val==val_predictions)/len(y_val)
#     val_f1 = f1_score(y_val, val_predictions, average='macro')

#     print('Train:      Accuracy %.4f | F1 %.4f \nValidation: Accuracy %.4f | F1 %.4f'%
#     (train_acc,train_f1,val_acc,val_f1))
#     print('------------------------------'*2)
#     print('------------------------------'*2)
    
#     random_params['train_acc'] = round(train_acc,4)
#     random_params['train_f1'] = round(train_f1, 4)
#     random_params['val_acc'] = round(val_acc, 4)
#     random_params['val_f1'] = round(val_f1, 4)
    
#     random_grid_scores.append(random_params)


# grid_search_results = pd.DataFrame(random_grid_scores)
# grid_search_results

# grid_search_results.sort_values('val_f1')

#####################################################################################

results_columns = [dataset + '_' + metric for dataset in ['Train', 'Val', 'Test'] 
                                          for metric in ['Acc', 'F1', 'MSE', 'Precision', 'Recall']] + ['Time', 'Memory']
rerun_results = []
for i in range(3):

    np.random.seed(20+i)

    rf = RandomForestClassifier(500,
                                criterion='gini',
                                min_samples_split=2,
                                max_depth=None,
                                max_features=27,
                                max_samples=None,
                                n_jobs=max(1,cpus-2), 
                                verbose=0)
    # memory 
    ! free -m | grep Mem | awk '{print $3}' > non_dl_memory.txt

    start= time.time()
    rf.fit(X_train, y_train)
    ! free -m | grep Mem | awk '{print $3}' >> non_dl_memory.txt
    end = time.time()-start
    minutes = end // 60
    seconds = end % 60
    print('Time: %i minutes %i seconds'%(minutes, seconds))

    # predict on train 
    train_predictions =  rf.predict(X_train)
    train_acc = np.sum(y_train==train_predictions)/len(y_train)
    train_f1 = f1_score(y_train, train_predictions, average='macro')
    train_precision = precision_score(y_train, train_predictions, average='macro')
    train_recall = recall_score(y_train, train_predictions, average='macro')
    train_mse = mean_squared_error(y_train, train_predictions)

    # Linux command, adjust if necessary to other OS or 
    # search and remove all memory variables
    ! free -m | grep Mem | awk '{print $3}' >> non_dl_memory.txt

    # predict on validation
    val_predictions = rf.predict(X_val)
    val_acc = np.sum(y_val==val_predictions)/len(y_val)
    val_f1 = f1_score(y_val, val_predictions, average='macro')
    val_precision = precision_score(y_val, val_predictions, average='macro')
    val_recall = recall_score(y_val, val_predictions, average='macro')
    val_mse = mean_squared_error(y_val, val_predictions)

    # Linux command, adjust if necessary to other OS
    ! free -m | grep Mem | awk '{print $3}' >> non_dl_memory.txt

    print('Train:      Accuracy %.4f | F1 %.4f \nValidation: Accuracy %.4f | F1 %.4f'%
    (train_acc,train_f1,val_acc,val_f1))
    print('------------------------------'*2)
    print('------------------------------'*2)

    # predict on test
    test_predictions = rf.predict(X_test)
    test_acc = np.sum(y_test==test_predictions)/len(y_test)
    test_f1 = f1_score(y_test, test_predictions, average='macro')
    test_precision = precision_score(y_test, test_predictions, average='macro')
    test_recall = recall_score(y_test, test_predictions, average='macro')    
    test_mse = mean_squared_error(y_test, test_predictions)

    # Linux command, adjust if necessary to other OS
    ! free -m | grep Mem | awk '{print $3}' >> non_dl_memory.txt

    memory_usage = np.loadtxt('non_dl_memory.txt', dtype='int', delimiter = '\n') 
    max_memory_usage = int(np.max(memory_usage))


    total_time = time.time()-start

    non_dl_run_results = [train_acc, train_f1, train_mse, train_precision, train_recall, 
                          val_acc, val_f1, val_mse, val_precision, val_recall,
                          test_acc, test_f1, test_mse, test_precision, test_recall,
                          total_time, max_memory_usage]
    
    rerun_results.append(non_dl_run_results)

    dump(rf, f'rf_classifier_run_{i+1}.joblib') 

raw_results = pd.DataFrame(rerun_results, columns=results_columns)

rf_benchmark_results = np.round(np.mean(raw_results[['Train_Acc', 'Train_F1', 'Train_MSE', 'Val_Acc', 'Val_F1', 'Val_MSE', 'Test_Acc', 'Test_F1', 'Test_MSE']], axis=0),4)
rf_benchmark_std = np.round(np.std(raw_results[['Train_Acc', 'Train_F1', 'Train_MSE', 'Val_Acc', 'Val_F1', 'Val_MSE', 'Test_Acc', 'Test_F1', 'Test_MSE']], axis=0),4)


rf_memory = np.max(raw_results['Memory'])
rf_time = round(np.mean(raw_results['Time'])/60,2)

