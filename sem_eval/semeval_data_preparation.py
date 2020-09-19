###### data preparation of SemEval-2019 dataset
import os
import time
import pandas as pd 
import transformers
import torch

semeval_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(semeval_path)

### Loading SemEval 2019 data
semeval_data = pd.read_csv('semeval_data.tsv',sep='\t', 
                           names=['hyperpartisan_label', 'unknown1', 'unknown2','content', 'title'])


# removing <splt> from the beginning of paragraphs
contents = semeval_data['content']
contents = [article.replace('<splt>', '') for article in contents]

titles = semeval_data['title']

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

### tokenize titles
start = time.time()
titles_tokenized = [bert_tokenizer.tokenize(item) for item in titles]
title_tokenization_time = time.time() - start
print('Tokenizing titles took ' + str(title_tokenization_time) + ' sec')

### tokenize contents 
start = time.time()
contents_tokenized = [bert_tokenizer.tokenize(item) for item in contents]
content_tokenization_time = time.time() - start
print('Tokenizing contents took ' + str(round(content_tokenization_time/60,2)) + ' min')

### converting tokens to ids, truncate, pad, and create masks

#########################
name_string = 'semeval_'#
#########################

# for titles
truncation_value =  50
titles_text = []
titles_mask = []

for item in titles_tokenized:
    output_dict = bert_tokenizer.encode_plus(item, 
                                                max_length= truncation_value,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=False)
    titles_text.append(output_dict['input_ids'])
    titles_mask.append(output_dict['attention_mask'])
# convert to pytorch tensor
titles_text_tensor = torch.tensor(titles_text)
titles_mask_tensor = torch.tensor(titles_mask)

# for contents
truncation_value = 500
contents_text = []
contents_mask = []
for item in contents_tokenized:
    output_dict = bert_tokenizer.encode_plus(item,
                                                max_length=truncation_value,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_token_type_ids=False)
    contents_text.append(output_dict['input_ids'])
    contents_mask.append(output_dict['attention_mask'])
# convert to pytorch tensor
contents_text_tensor = torch.tensor(contents_text)
contents_mask_tensor = torch.tensor(contents_mask)

# create bias tensor
bias_tensor = torch.Tensor(semeval_data['hyperpartisan_label']*1)

### save tensors
path = semeval_path

torch.save(titles_text_tensor,f'{path}{name_string}titles_text_tensor.pt')
torch.save(titles_mask_tensor,f'{path}{name_string}titles_mask_tensor.pt')

torch.save(contents_text_tensor,f'{path}{name_string}contents_text_tensor.pt')
torch.save(contents_mask_tensor,f'{path}{name_string}contents_mask_tensor.pt')

torch.save(bias_tensor,f'{path}{name_string}bias_tensor.pt')











