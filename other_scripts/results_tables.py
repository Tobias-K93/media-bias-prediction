############### Results Tables ###############
import pandas as pd
import numpy as np
import os 

# set path to media-bias-prediction repository 
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(repo_path, 'deep_learning_models'))

file_names = ['full_without_wrongly_labeled', 'aggregators_removed', 'tabloids_removed', 
              'duplicates_removed', 
              'aggregators_tabloids_duplicates_removed']

### Applied datasets results
wanted_results = []
for name in file_names:
    score_list = []
    for i in range(3):
        scores = pd.read_csv(os.path.join('scores', f'metric_scores_allsides_{name}_rerun_{i+1}.csv')).iloc[-1,:]
        score_list.append(scores)
    wanted_results += score_list

applied_datasets_each_run = pd.DataFrame(wanted_results).drop(columns=['epoch', 'time', 'train_precision', 'train_recall',
                                                 'val_precision', 'val_recall', 'test_precision',
                                                 'test_recall', 'train_loss', 'val_loss', 'test_loss',
                                                 'memory']).__round__(4)


standard_deviations = np.zeros((5,9))
for i in range(0,13,3):
    std_array = np.std(applied_datasets_each_run.iloc[i:i+3,:],axis=0, ddof=1)
    standard_deviations[int(i/3),:] = std_array.round(4)

applied_datasets_std = pd.DataFrame(standard_deviations, columns=applied_datasets_each_run.columns)

# move latex function from back to beginning if needed
#latex_output_fct(applied_datasets_std)

average_results = []
for i in range(0,len(wanted_results),3):
    average_results.append((wanted_results[i+0]+wanted_results[i+1]+wanted_results[i+2])/3)


average_results_df = pd.DataFrame(average_results, index=file_names).__round__(4)

final_results = average_results_df.drop(columns=['epoch', 'time', 'train_precision', 'train_recall',
                                                 'val_precision', 'val_recall', 'test_precision',
                                                 'test_recall', 'train_loss', 'val_loss', 'test_loss',
                                                 'memory'])


### SHA-BiLSTM benchmark scores
dl_score_list = []
for i in range(3):
    scores = pd.read_csv(os.path.join('dl_benchmark_scores', f'metric_scores_dl_benchmark_allsides_all_removed_rerun_{i+1}.csv')).iloc[-1,:]
    dl_score_list.append(scores)

dl_score_df = pd.DataFrame(dl_score_list).drop(columns=['epoch', 'time', 'train_precision', 'train_recall', 
                                                        'val_precision', 'val_recall', 'test_precision',
                                                        'test_recall', 'train_loss', 'val_loss', 'test_loss',
                                                        'memory'])

dl_standard_deviations = np.std(dl_score_df,axis=0).round(4)

dl_average_results = np.mean(dl_score_df, axis=0).__round__(4)

#latex_output_fct(dl_standard_deviations)

### Benchmark time and memory results

# Bert
bert_time_memory_df = pd.DataFrame(columns=['time','memory'])
for i in range(3):
    temp_df = pd.read_csv(os.path.join('scores', f'metric_scores_allsides_aggregators_tabloids_duplicates_removed_rerun_{i+1}.csv'))[['time','memory']]
    bert_time_memory_df = pd.concat([bert_time_memory_df,temp_df],)

bert_avg_time = round(np.sum(bert_time_memory_df['time'])/3,2)
bert_max_memory = np.max(bert_time_memory_df['memory'])

# SHA-BiLSTM
# batch=64
bilstm_time_memory_df = pd.DataFrame(columns=['time','memory'])
for i in range(3):
    temp_df = pd.read_csv(os.path.join('dl_benchmark_scores', f'metric_scores_dl_benchmark_allsides_all_removed_rerun_{i+1}.csv'))[['time','memory']]
    bilstm_time_memory_df = pd.concat([bilstm_time_memory_df,temp_df],)

bilstm_avg_time = round(np.sum(bilstm_time_memory_df['time'])/3,2)
bilstm_max_memory = np.max(bilstm_time_memory_df['memory'])

# batch=16
bilstm16_df = pd.read_csv(os.path.join('dl_benchmark_scores', f'metric_scores_dl_benchmark_allsides_batch_16_all_removed_rerun_1.csv'))[['time','memory']]
bilstm16_avg_time = round(np.sum(bilstm16_df['time']),2) # /3
bilstm16_max_memory = np.max(bilstm16_df['memory'])

### Cost sensitive results
cost_sensitive_score_list = []
for i in range(3):
    scores = pd.read_csv(os.path.join('scores', f'metric_scores_allsides_cost_sensitive_all_removed_rerun_{i+1}.csv')).iloc[-1,:]
    cost_sensitive_score_list.append(scores)

cost_sensitive_score_df = pd.DataFrame(cost_sensitive_score_list).drop(columns=['epoch', 'time', 'train_precision', 'train_recall',
                                                    'val_precision', 'val_recall', 'test_precision',
                                                    'test_recall', 'train_loss', 'val_loss', 'test_loss',
                                                     'memory'])

cost_sensitive_standard_deviations = np.std(cost_sensitive_score_df, axis=0).round(4)

cost_sensitive_average_results = np.mean(cost_sensitive_score_df,axis=0).round(4)

#latex_output_fct(cost_sensitive_standard_deviations)


### Excluded sources results
excluded_sources_results = []
excluded_sources_std = []
for group,sources_in_training in zip(['small', 'small', 'large', 'large'],
                                     ['with_sources', 'without_sources','with_sources', 'without_sources']):

    results_per_category_list = []
    for run in range(1,4):
        single_run_df = pd.read_csv(os.path.join('scores', f'accuracy_scores_{group}_{sources_in_training}_run_{run}.csv')).iloc[0,:]
        results_per_category_list.append(single_run_df)
    
    results_per_category_df = pd.DataFrame(results_per_category_list)
    average_per_category = np.mean(results_per_category_df, axis=0)
    std_per_category = np.std(results_per_category_df, axis=0)

    excluded_sources_results.append(average_per_category)
    excluded_sources_std.append(std_per_category)

excluded_sources_small = pd.DataFrame(excluded_sources_results[:2]).__round__(4)
excluded_sources_small
excluded_sources_large = pd.DataFrame(excluded_sources_results[2:]).__round__(4)
excluded_sources_large

excluded_sources_small_std = pd.DataFrame(excluded_sources_std[:2]).__round__(4)
excluded_sources_large_std = pd.DataFrame(excluded_sources_std[2:]).__round__(4)

#latex_output_fct(excluded_sources_small_std)
#latex_output_fct(excluded_sources_large_std)

### SemEval 
# directly out of notebook

# def latex_output_fct(data_frame, std=True):
#     if std == True:
#         try:
#             for _, row in data_frame.iterrows():
#                 latex_string = ''
#                 for i,value in enumerate(list(row)):
#                     if i == len(row)-1:
#                         latex_string += '\scriptsize (' + str(value) + ')' +' \\\\'
#                     elif i==0:
#                         latex_string += '& \scriptsize (' + str(value) + ') & '
#                     else:
#                         latex_string += '\scriptsize (' + str(value) + ') & '
#                 print(latex_string)

#         except AttributeError:
#             row = data_frame
#             latex_string = ''
#             for i,value in enumerate(list(row)):
#                 if i == len(row)-1:
#                     latex_string += '\scriptsize (' + str(value) + ')' +' \\\\'
#                 elif i==0:
#                     latex_string += '& \scriptsize (' + str(value) + ') & ' 
#                 else:
#                     latex_string += '\scriptsize (' + str(value) + ') & '
#             print(latex_string)
#     else:
#         try:
#             for _, row in data_frame.iterrows():
#                 latex_string = ''
#                 for i,value in enumerate(list(row)):
#                     if i == len(row)-1:
#                         latex_string += str(value) +' \\\\'
#                     else:
#                         latex_string += str(value) + ' & '
#                 print(latex_string)

#         except AttributeError:
#             row = data_frame
#             latex_string = ''
#             for i,value in enumerate(list(row)):
#                 if i == len(row)-1:
#                     latex_string += str(value) +' \\\\'
#                 else:
#                     latex_string += str(value) + ' & '
#             print(latex_string)
