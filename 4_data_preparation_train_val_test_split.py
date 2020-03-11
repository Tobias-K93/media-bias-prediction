########## Creating Test Set ##########
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

### loading tensors
contents_text_tensor = torch.load('contents_text_tensor_source_removed.pt')
contents_mask_tensor = torch.load('contents_mask_tensor_source_removed.pt')

titles_text_tensor = torch.load('titles_text_tensor.pt')
titles_mask_tensor = torch.load('titles_mask_tensor.pt')

bias_tensor = torch.load('bias_tensor.pt')
### loading date and source
data = pd.read_csv('/home/tobias/Documents/Studium/Master_thesis/programming/data_short.csv')
data.drop(columns=['name', 'content', 'bias'],inplace=True)

### Converting source to id array with dict ###
source_array = np.array(data['source']).reshape((-1,1))
# initilizing encoder and convert source array
source_encoder = OrdinalEncoder(dtype=np.int8)
source_encoder.fit(source_array)
source_transformed = source_encoder.transform(source_array)
# create dictionary for later 
source_dict = {}
for i,source in enumerate(source_encoder.categories_[0]):
    source_dict[source] = i

source_dict_inverse = {}
for i,source in enumerate(source_encoder.categories_[0]):
    source_dict_inverse[i] = source

###############################################

### creating id vectors
np.random.seed(123)

ids = np.arange(len(bias_tensor))
np.random.shuffle(ids) 
# cut offs for train- validation- and test-set 80%/10%/10%
train_val_cut = int(len(bias_tensor)*0.8)
val_test_cut = int(len(bias_tensor)*0.9)
# ids for each set
train_ids = ids[:train_val_cut]
val_ids = ids[train_val_cut:val_test_cut]
test_ids = ids[val_test_cut:]

### creating train- val- test-sets
tensor_list = [contents_text_tensor, contents_mask_tensor, titles_text_tensor, titles_mask_tensor, bias_tensor, source_transformed]
tensor_file_names = ['contents_text_source_removed', 'contents_mask_source_removed', 'titles_text', 'titles_mask', 'bias', 'source']

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
path = '/home/tobias/Documents/Studium/Master_thesis/programming/ready_to_use/'

for tensor,name in zip(train_tensors,tensor_file_names):
    if name == 'source':
        np.save(path + name + '_train.npy', tensor)
    else:
        torch.save(tensor, path + name + '_train.pt')

for tensor,name in zip(val_tensors,tensor_file_names):
    if name == 'source':
        np.save(path + name + '_val.npy', tensor)
    else:
        torch.save(tensor, path + name + '_val.pt')

for tensor,name in zip(test_tensors,tensor_file_names):
    if name == 'source':
        np.save(path + name + '_test.npy', tensor)
    else:
        torch.save(tensor, path + name + '_test.pt')

