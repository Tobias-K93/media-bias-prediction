########### Non-DL Benchmark model ##########
import time
import multiprocessing
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid
############################
### choosing split ratio ###
split_ratio = (60,20,20) ###
############################

### loading target variable
y = torch.load('bias_tensor.pt').numpy()

### loading linguistic variables
character_variables = np.load('non_dl_character_variables.npy')
dict_count_variables = np.load('non_dl_dict_count_variables.npy')
token_based_variables = np.load('non_dl_token_based_variables.npy')
pos_variables = np.load('non_dl_pos_variables.npy')

X = np.concatenate((character_variables,dict_count_variables,
                    token_based_variables, pos_variables),axis=1)





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
cpus = 5 #multiprocessing.cpu_count()

### hyper parameter search

parameters = {'n_estimators': [200,500],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None,20,30,40,50],
              'min_samples_split': [10,20,30,50,100] ,
              'max_features': ['sqrt', None]}

param_grid = ParameterGrid(parameters)

grid_search_iterations = 10

random_grid_ids = np.random.randint(0,len(param_grid)-1,grid_search_iterations)

random_grid_scores = []
for i,id in enumerate(random_grid_ids):
    random_params = param_grid[id]

    rf = RandomForestClassifier(n_estimators=random_params['n_estimators'],
                                criterion= random_params['criterion'],
                                min_samples_split=random_params['min_samples_split'],
                                max_features= random_params['max_features'],
                                max_depth= random_params['max_depth'],
                                n_jobs=max(1,cpus-1), 
                                verbose=0)

    print(pd.DataFrame(random_params, index=[i+1]))
    print('------------------------------'*2)
    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()-start
    minutes = end // 60
    seconds = end % 60
    print('Time: %i minutes %i seconds'%(minutes, seconds))

    # predict on train 
    train_predictions =  rf.predict(X_train)
    train_acc = np.sum(y_train==train_predictions)/len(y_train)
    train_f1 = f1_score(y_train, train_predictions, average='macro')

    # predict on validation
    val_predictions = rf.predict(X_val)
    val_acc = np.sum(y_val==val_predictions)/len(y_val)
    val_f1 = f1_score(y_val, val_predictions, average='macro')

    print('Train:      Accuracy %.4f | F1 %.4f \nValidation: Accuracy %.4f | F1 %.4f'%
    (train_acc,train_f1,val_acc,val_f1))
    print('------------------------------'*2)
    print('------------------------------'*2)
    
    random_params['train_acc'] = round(train_acc,4)
    random_params['train_f1'] = round(train_f1, 4)
    random_params['val_acc'] = round(val_acc, 4)
    random_params['val_f1'] = round(val_f1, 4)
    
    random_grid_scores.append(random_params)


grid_search_results = pd.DataFrame(random_grid_scores)
grid_search_results



#####################################################################################

rf = RandomForestClassifier(200,
                            min_samples_leaf=100,
                            n_jobs=max(1,cpus-1), 
                            verbose=1)

start= time.time()
rf.fit(X_train, y_train)
print('RandomnForest took %.2f seconds'%(time.time()-start))

train_predictions =  rf.predict(X_train)
train_acc = np.sum(y_train==train_predictions)/len(y_train)
train_f1 = f1_score(y_train, train_predictions, average='macro')

### Predict on validation

val_predictions = rf.predict(X_val)
val_acc = np.sum(y_val==val_predictions)/len(y_val)
val_f1 = f1_score(y_val, val_predictions, average='macro')

print('Train: Accuracy %.4f | F1 %.4f \nValidation: Accuracy %.4f | F1 %.4f'%
(train_acc,train_f1,val_acc,val_f1))






