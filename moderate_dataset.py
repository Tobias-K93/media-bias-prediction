############### Moderate dataset ###############
from data_preparation_fct import data_preparation
import pandas as pd
import numpy as np
import torch

# load data
data = pd.read_csv('moderate_bias/moderate_articles.csv')

bias_dict = {'moderate_left': 0, 'moderate_right': 1}
data_preparation(data, 'moderate', bias_dict)

data_short = pd.read_csv('moderate_bias/moderate_data_short.csv')

source = np.array(data_short['source'])

np.save('moderate_bias/moderate_sources.npy',source)


### split into validation- and test-set
contents_mask_tensor = torch.load('moderate_bias/moderate_contents_mask_tensor.pt')
contents_text_tensor = torch.load('moderate_bias/moderate_contents_text_tensor.pt')
bias_tensor = torch.load('moderate_bias/moderate_bias_tensor.pt')
#source = np.load('moderate_bias/moderate_sources.npy', allow_pickle=True)


tensor_list = [contents_text_tensor, contents_mask_tensor, bias_tensor, source] # titles_text_tensor, titles_mask_tensor
tensor_file_names = ['contents_text', 'contents_mask', 'bias', 'source'] # 'titles_text', 'titles_mask'

np.random.seed(123)

data_length = len(bias_tensor) 

ids = np.arange(data_length)
np.random.shuffle(ids) 
# cut off for validation- and test-set 
cut = int(data_length*0.5)
# ids for each set
val_ids = ids[:cut]
test_ids = ids[cut:]


val_tensors = []
for tensor in tensor_list:
    val_tensors.append(tensor[val_ids])

test_tensors = []
for tensor in tensor_list:
    test_tensors.append(tensor[test_ids])

### saving tensors

affix = 'moderate_'

path = '/home/tobias/Documents/Studium/Master_thesis/programming/moderate_bias/'

for tensor,name in zip(val_tensors,tensor_file_names):
    if name == 'source':
        np.save(path + affix + name +'_val.npy', tensor)
    else:
        torch.save(tensor, path + affix + name + '_val.pt')

for tensor,name in zip(test_tensors,tensor_file_names):
    if name == 'source':
        np.save(path + affix + name +'_test.npy', tensor)
    else:
        torch.save(tensor, path + affix + name + '_test.pt')




