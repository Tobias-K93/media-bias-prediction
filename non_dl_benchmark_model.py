########### Non-DL Benchmark model ##########
import time
import multiprocessing
import numpy as np
import torch
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

X = np.concatenate((character_variables,dict_count_variables,token_based_variables),axis=1)


### creating id vectors
np.random.seed(123)
data_length = len(y) #len(tensor_cut_sun_list[0])

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

parameters = {'n_estimators': [100,200,300,500],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 5,7,10,14],
              'min_samples_split': [10,30,50,100,200] ,
              'max_features': ['sqrt', None]}

param_grid = ParameterGrid(parameters)

grid_search_iterations = 10

random_grid_ids = np.random.randint(0,len(param_grid)-1,grid_search_iterations)

for id in random_grid_ids:
    

    rf = RandomForestClassifier(n_estimators=,
                                min_samples_leaf=100,
                                n_jobs=max(1,cpus-1), 
                                verbose=1)








rf = RandomForestClassifier(200,
                            min_samples_leaf=100,
                            n_jobs=max(1,cpus-1), 
                            verbose=1)

start= time.time()
rf.fit(X_train, y_train)
print('RandomnForest took %f.2 seconds'%(time.time()-start))

train_predictions =  rf.predict(X_train)
train_acc = np.sum(y_train==train_predictions)/len(y_train)
train_f1 = f1_score(y_train, train_predictions, average='macro')

### Predict on validation

val_predictions = rf.predict(X_val)
val_acc = np.sum(y_val==val_predictions)/len(y_val)
val_f1 = f1_score(y_val, val_predictions, average='macro')

print('Train: Accuracy %f.4 | F1 %f.4 \nValidation: Accuracy %f.4 | F1 %f.4'%
(train_acc,train_f1,val_acc,val_f1))






