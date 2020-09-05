########## Preparing data for BERT ##########
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch
#################################
affix = ''                    ### '_duplicates_removed'
#################################

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

### reading id files
# with open("titles_ids.csv", newline='') as f:
#     reader = csv.reader(f)
#     titles_ids = [[int(item) for item in row] for row in reader]

with open(f"contents_ids{affix}.csv", newline='') as f:
    reader = csv.reader(f)
    contents_ids = [[int(item) for item in row] for row in reader]

### truncate, pad and add special tokens
# for titles
# titles_lengths = [len(item) for item in titles_ids]

# truncation_value = max(titles_lengths) # 50
# titles_text = []
# titles_mask = []
# for item in titles_ids:
#     dict_values = tuple(bert_tokenizer.prepare_for_model(item,
#                                         max_length=truncation_value,
#                                         pad_to_max_length=True,
#                                         return_attention_mask=True,
#                                         return_token_type_ids=False).values())
#     titles_text.append(dict_values[0])
#     titles_mask.append(dict_values[1])

# for contents
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

# convert to pytorch tensor
# titles_text_tensor = torch.tensor(titles_text)
# titles_mask_tensor = torch.tensor(titles_mask)

contents_text_tensor = torch.tensor(contents_text)
contents_mask_tensor = torch.tensor(contents_mask)

# save tensors
# torch.save(titles_text_tensor,'titles_text_tensor.pt')
# torch.save(titles_mask_tensor,'titles_mask_tensor.pt')

torch.save(contents_text_tensor,f'contents_text_tensor{affix}.pt')
torch.save(contents_mask_tensor,f'contents_mask_tensor{affix}.pt')
