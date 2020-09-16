#################### Random Forest SemEval evaluation ####################
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

### Loading SemEval 2019 variables
character_variables = np.load(f'non_dl_semeval_character_variables.npy')
dict_count_variables = np.load(f'non_dl_semeval_dict_count_variables.npy')
token_based_variables = np.load(f'non_dl_semeval_token_based_variables.npy')
pos_variables = np.load(f'non_dl_semeval_pos_variables.npy')

X = np.concatenate((character_variables,dict_count_variables,
                    token_based_variables, pos_variables),axis=1)

# 2 cases of Honore's R == inf, replace with high number instead
X[X==np.inf] = 10000

y = np.load(f'non_dl_semeval_bias.npy', allow_pickle=True)

results_per_run_list = []
for i in range(3):

    rf = load(f'rf_classifier_run_{i+1}.joblib')

    predictions = rf.predict(X)

    # converting left-right 5 level labels to hyper/ non-hyper 2 level labels
    y_binary = np.zeros(len(y))
    y_binary[y==0] = 1
    y_binary[y==1] = 0
    y_binary[y==2] = 0
    y_binary[y==3] = 0
    y_binary[y==4] = 1

    predictions_binary = np.zeros(len(predictions))
    predictions_binary[predictions==0] = 1
    predictions_binary[predictions==1] = 0
    predictions_binary[predictions==2] = 0
    predictions_binary[predictions==3] = 0
    predictions_binary[predictions==4] = 1


    acc = np.sum(y_binary==predictions_binary)/len(y_binary)
    f1 = f1_score(y_binary, predictions_binary)

    results_per_run_list.append([acc,f1])
    print(f'Accuracy: {acc:.4},    F1-score: {f1:.4}')


results_per_run_list_df = pd.DataFrame(results_per_run_list, columns=['Acc','F1'])

final_results = np.mean(results_per_run_list_df, axis=0).round(4)
results_std = np.std(results_per_run_list_df, axis=0).round(4)




