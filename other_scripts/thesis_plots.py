#################### Plots for Thesis ####################
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os 

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(repo_path, 'data_preparation'))


allsides_bias_tensor = torch.load(os.path.join('allsides_data', 'allsides_bias_tensor.pt'))
allsides_bias_array = allsides_bias_tensor.numpy()

mbfc_bias_tensor = torch.load(os.path.join('mbfc_data', 'mbfc_bias_tensor.pt'))
mbfc_bias_array = mbfc_bias_tensor.numpy()

allsides_palette = {'Left': 'blue', 'Lean Left': 'slateblue', 'Center': 'grey', 
                    'Lean Right': 'indianred', 'Right': 'red'}
mbfc_palette = {'Extreme Left': 'darkblue',
                'Left': 'blue',
                'Left Center': 'slateblue',
                'Least Biased': 'grey',
                'Right Center': 'indianred',
                'Right': 'red',
                'Extreme Right': 'darkred'}


mbfc_plot_labels = ['Extreme Left', 'Left', 'Left Center', 'Least Biased', 
                    'Right Center', 'Right', 'Extreme Right']

### bias labels ###########################################################################
allsides_label_distribution = np.unique(allsides_bias_array, return_counts=True)[1]

mbfc_dist_dict = {label: value for label, value in 
                  zip(np.unique(mbfc_bias_array, return_counts=True)[0],
                  np.unique(mbfc_bias_array, return_counts=True)[1])}
mbfc_label_distribution = [mbfc_dist_dict[label] for label in ['extreme_left', 'left_bias', 'left_center_bias', 'least_biased', 'right_center_bias', 'right_bias', 'extreme_right']]

plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(1,2, figsize=(18,7), sharey=False) 

axes[0].set_title('Allsides')
#axes.set_xlim([0,y_limit])
axes[0].set_ylabel('Bias')
axes[0].set_xlabel('Number of articles')
axes[0].grid(True)


axes[1].set_title('Media Bias/Fact Check')
# axes[1].set_xlim([0,y_limit])
axes[1].set_ylabel('Bias')
axes[1].set_xlabel('Number of articles')
axes[1].grid(True)

sns.barplot(allsides_label_distribution, list(allsides_palette.keys()), ax = axes[0], palette=allsides_palette, orient='h') 
sns.barplot(mbfc_label_distribution, mbfc_plot_labels, ax = axes[1], palette=mbfc_palette, orient='h') 

fig.tight_layout()

fig.savefig('allsides_and_mbfc_label_distribution.png')


### Sources ###########################################################################
allsides_sources = np.load(os.path.join('allsides_data', 'allsides_sources.npy'), allow_pickle=True).flatten()
allsides_bias_labels = pd.read_csv(os.path.join('allsides_data', 'allsides_bias_labels.csv'))

sources_names, sources_frequencies = np.unique(allsides_sources, return_counts=True)
allsides_frequency_df = pd.DataFrame(np.concatenate((sources_names.reshape(-1,1), sources_frequencies.reshape(-1,1)), axis=1), columns=['Source', 'frequency'])

allsides_bias_labels = allsides_bias_labels.merge(allsides_frequency_df, on='Source')

allsides_bias_labels_sorted = []
for bias_label in ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']:
    allsides_bias_labels_sorted.append(allsides_bias_labels[allsides_bias_labels['bias']==bias_label].sort_values('frequency'))
allsides_bias_labels_sorted = pd.concat(allsides_bias_labels_sorted)

wrongly_labeled = ['RightWingWatch']
news_aggregators = ['Drudge Report', 'Real Clear Politics', 'Yahoo News'] 
tabloids = ['New York Daily News', 'Daily Mail', 'New York Post']
 
for i,name in enumerate(allsides_bias_labels_sorted['Source']):
    if name in news_aggregators:
        new_name = allsides_bias_labels_sorted.iloc[i,0] + ' (A)'
        allsides_bias_labels_sorted.iloc[i,0] = new_name
    elif name in tabloids:
        new_name = allsides_bias_labels_sorted.iloc[i,0] + ' (T)'
        allsides_bias_labels_sorted.iloc[i,0] = new_name

frequency_sorted = list(allsides_bias_labels_sorted['frequency'])
names_sorted = list(allsides_bias_labels_sorted['Source'])

sources_palette = {source: allsides_palette[bias] for source, bias in zip(allsides_bias_labels_sorted['Source'], allsides_bias_labels_sorted['bias'])}

handles =  [plt.Rectangle((0,0),1,1, color=allsides_palette[label]) for label in allsides_palette.keys()]
legend_labels = allsides_palette.keys()

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(1,1, figsize=(12,16), sharey=False) 

ax.set_ylabel('Source')
ax.set_xlabel('Number of articles')
ax.grid(True)
# ax.set_xscale('log') #, subsx=[2, 3, 4, 5, 6, 7, 8, 9]
# ax.set_xlim((10,100000))
#ax.tick_params(width=2)

ax.legend(handles, legend_labels, ncol=5, bbox_to_anchor=(1, 1.05)) # loc='upper left', 

sns.barplot(frequency_sorted, names_sorted , ax=ax, palette = sources_palette, orient='h')
fig.tight_layout()

fig.savefig('news_sources_distribution.png')

