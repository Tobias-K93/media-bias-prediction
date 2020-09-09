########## Preparing data for BERT ##########
import time
import csv
import os
import numpy as np
import pandas as pd
import transformers
import torch


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'allsides_data')
os.chdir(data_path)

### Select name of dataset to name files
####################################################
affix = 'allsides' # 'allsides_duplicates_removed' # 
####################################################

if (affix == 'allsides') | (affix == 'allsides_duplicates_removed'):
    pass
else:
    raise AssertionError('affix should be \'allsides\' or \'allsides_duplicates_removed\'')

### load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

### create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

##### contents
### reading id files
with open(f"{affix}_contents_ids.csv", newline='') as f:
    reader = csv.reader(f)
    contents_ids = [[int(item) for item in row] for row in reader]

### truncate, pad and add special tokens
truncation_value = 500
contents_text = []
contents_mask = []
for item in contents_ids:
    dict_values = tuple(bert_tokenizer.prepare_for_model(item,
                                        max_length=truncation_value,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_token_type_ids=False).values())
    contents_text.append(dict_values[0])
    contents_mask.append(dict_values[1])

contents_text_tensor = torch.tensor(contents_text)
contents_mask_tensor = torch.tensor(contents_mask)

### save tensors
torch.save(contents_text_tensor,f'{affix}_contents_text_tensor.pt')
torch.save(contents_mask_tensor,f'{affix}_contents_mask_tensor.pt')

##### titles (not used in thesis)
### reading id files
# with open(f"{affix}_titles_ids.csv", newline='') as f:
#     reader = csv.reader(f)
#     titles_ids = [[int(item) for item in row] for row in reader]

### truncate, pad and add special tokens
# titles_lengths = [len(item) for item in titles_ids]
# titles_lengths_quantiles = np.quantile(titles_lengths, [0.5,0.9,0.95]) 

# # truncation value is equivalent to 95% quantile
# truncation_value = 20 
# titles_text = []
# titles_mask = []
# for item in titles_ids:
#     dict_values = tuple(bert_tokenizer.prepare_for_model(item,
#                                         max_length=truncation_value,
#                                         add_special_tokens=False,
#                                         pad_to_max_length=True,
#                                         return_attention_mask=True,
#                                         return_token_type_ids=False).values())
#     titles_text.append(dict_values[0])
#     titles_mask.append(dict_values[1])

# # convert to pytorch tensor
# titles_text_tensor = torch.tensor(titles_text)
# titles_mask_tensor = torch.tensor(titles_mask)

### save tensors
# torch.save(titles_text_tensor,f'{affix}_titles_text_tensor.pt')
# torch.save(titles_mask_tensor,f'{affix}_titles_mask_tensor.pt')

