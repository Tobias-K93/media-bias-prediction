########## Creating Test Set ##########
import torch
import numpy as np
import pandas as pd
import os

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'allsides_data')
os.chdir(data_path)

### Select name of dataset to name files
####################################################
affix = 'allsides' # 'allsides_duplicates_removed' # 
####################################################
# (for allsides_duplicates_removed only texts and masks 
#  need to be handled, since rest is the same)
### choosing split ratio ###
split_ratio = (80,10,10) ###
############################

##### loading tensors
### contents
contents_text_tensor = torch.load(f'{affix}_contents_text_tensor.pt')
contents_mask_tensor = torch.load(f'{affix}_contents_mask_tensor.pt')

### titles
# titles_text_tensor = torch.load(f'{affix}_titles_text_tensor.pt')
# titles_mask_tensor = torch.load(f'{affix}_titles_mask_tensor.pt')

bias_tensor = torch.load(f'{affix}_bias_tensor.pt')

### loading date and source
data = pd.read_csv('allsides_data_short.csv')
data.drop(columns=['name', 'content', 'bias'],inplace=True)

source_array = np.array(data['source']).reshape((-1,1))

# list of tensors that need to be: modified, devided into sets, and saved
if affix == 'allsides':
    tensor_list = [contents_text_tensor, contents_mask_tensor, bias_tensor, source_array]
elif affix == 'allsides_duplicates_removed':
    tensor_list = [contents_text_tensor, contents_mask_tensor]
else:
    raise AssertionError('affix should be \'allsides\' or \'allsides_duplicates_removed\'')
# titles_text_tensor, titles_mask_tensor, # (titles not used)

### creating id vectors
np.random.seed(123)
data_length = len(contents_text_tensor) 

ids = np.arange(data_length)
np.random.shuffle(ids) 
# cut offs for train- validation- and test-set according to split ratio
train_val_cut = int(data_length*(split_ratio[0]/100))
val_test_cut = int(data_length*(split_ratio[0]+split_ratio[1])/100)
# ids for each set
train_ids = ids[:train_val_cut]
val_ids = ids[train_val_cut:val_test_cut]
test_ids = ids[val_test_cut:]

### creating train- val- test-sets
if affix == 'allsides':
    tensor_file_names = ['contents_text', 'contents_mask', 'bias', 'source']
elif affix == 'allsides_duplicates_removed':
    tensor_file_names = ['contents_text', 'contents_mask']
else:
    raise AssertionError('affix should be \'allsides\' or \'allsides_duplicates_removed\'')
# 'titles_text', 'titles_mask'

train_tensors = []
for tensor in tensor_list:
    train_tensors.append(tensor[train_ids])

val_tensors = []
for tensor in tensor_list:
    val_tensors.append(tensor[val_ids])

test_tensors = []
for tensor in tensor_list:
    test_tensors.append(tensor[test_ids])

### saving tensors
for tensor,name in zip(train_tensors,tensor_file_names):
    if name == 'source':
        np.save(f'{affix}_{name}_train.npy', tensor)
    else:
        torch.save(tensor,f'{affix}_{name}_train.pt')

for tensor,name in zip(val_tensors,tensor_file_names):
    if name == 'source':
        np.save(f'{affix}_{name}_val.npy', tensor)
    else:
        torch.save(tensor, f'{affix}_{name}_val.pt')

for tensor,name in zip(test_tensors,tensor_file_names):
    if name == 'source':
        np.save(f'{affix}_{name}_test.npy', tensor)
    else:
        torch.save(tensor, f'{affix}_{name}_test.pt')

