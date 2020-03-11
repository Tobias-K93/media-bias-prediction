########## data preparation for news bias dataset ##########
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch
# load data
data = pd.read_csv('bias_articles.csv')

# create lists of titles and contents
titles = list(data['name'])
contents = list(data['content'])

# ID's of entries without title
titles_nans_id = [i for i,item in enumerate(titles) if pd.isna(item)]

# ID's of entries without content 
contents_nans_id = [i for i,item in enumerate(contents) if pd.isna(item)]
# Sources of articles without content and the corresponding frequencies
np.unique(data['source'].iloc[contents_nans_id], return_counts=True)

# removing entries with nans 
nans_id = list(set(titles_nans_id+contents_nans_id))
data.drop(index=nans_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

# reset titles and contents lists
titles = list(data['name'])
contents = list(data['content'])

# get lengths of titles and contents 
titles_length = [len(item) for item in titles]
contents_length = [len(item) for item in contents]

### some stats on length in characters
titles_length_max = max(titles_length)
titles_length_quantiles = np.quantile(titles_length,[0.25,0.5,0.75])
# histogram of lengths of titles
# plt.hist(titles_length)

contents_length_max = max(contents_length)
contents_length_quantiles = np.quantile(contents_length,[0.25,0.5,0.75])

#########################################################################

##### Closer look at articles around the cutoff
# List of articles with wanted length
contents_length_array = np.array(contents_length)
list(np.arange(len(contents_length_array))[(contents_length_array > 900)*(contents_length_array < 1300)])

# It seems that most articles with little more than 500 characters
# are not real articles but either short summaries or short 
# snippets of longer articles, videos, and podcasts. It also 
# happens that they list topics of the the webside that are 
# discribed with one sentence each. 

print(data['source'].iloc[104])
print(contents[104])
#########################################################################

# large number of under 100 character length 
# plt.hist(contents_length,bins=50,range=(0,1500))
# plt.show

# large amount of articles with character length of 33 due to error message
# from source Raw Story '# 403 Forbidden\r\n\r\n* * *\r\n\r\nnginx' which 
# leads to 3633 out of 3719 articles that are not usable
# a lot of values around 60 seem to be due to short snippets out of original
# articles by e.g. New York Post

### distribution of uncut articles' lengths
# plt.hist(contents_length,bins=39,range=(500,20000))
# plt.show

#########################
# Among the very short articles is a lot of noise (non-news content, 
# short description linking to full articles, error messages,...). Thus, removing articles
# which are uncommenly short (under 500 characters ~ 100 words) should 
# reduce noise. Similarly very long articles (over 20000 characters ~ 4000 
# words) contain unwanted and rather rare cases e.g. several whole articles, 
# that belong to a specific topic, listed as one. 

# 38676 short cut
short_cut = np.sum(np.array(contents_length)<500)
# 1292 long cut
long_cut = np.sum(np.array(contents_length)>20000)

contents_unwanted_length_id = [i for i,item in enumerate(contents_length) if item < 500 or item > 20000]

# removing short and long articles from dataset
data.drop(index=contents_unwanted_length_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

##### Saving shortened dataset 
data.to_csv('data_short.csv', index=False)

###### create target labels vector #####
bias_dict = {'left': 0, 'center': 1, 'right': 2}

bias_list = [bias_dict[item] for item in data['bias']]
        
bias_tensor = torch.tensor(bias_list)

torch.save(bias_tensor,'bias_tensor.pt')
# removing saved bias objects
try:
    del(bias_list,bias_tensor)
except NameError:
    pass
########################################

# reset titles and contents lists
titles = list(data['name'])
contents = list(data['content'])

# Shortening contents to max 5000 characters (for faster tokenization)
contents_short = [item[:5000] for item in contents]
# delete long contents
try:
    del(contents)
except NameError:
    pass

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

########################################
### Tokenize sources for later removal from content
sources = list(data['source'])

start = time.time()
sources_tokenized = [bert_tokenizer.tokenize(item) for item in sources]
source_tokenization_time = time.time() - start
print('Tokenizing sources took ' + str(source_tokenization_time) + ' sec')

### writing tokenized sources list to csv file
with open("sources_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(sources_tokenized)

### delete saved sources objects
try:
    del(sources, sources_tokenized)
except NameError:
    pass

########################################

### tokenize titles
start = time.time()
titles_tokenized = [bert_tokenizer.tokenize(item) for item in titles]
title_tokenization_time = time.time() - start
print('Tokenizing titles took ' + str(title_tokenization_time) + ' sec')

### writing tokenized titles list to csv file
with open("titles_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(titles_tokenized)

### delete saved titles objects
try:
    del(titles, titles_tokenized)
except NameError:
    pass

# tokenize contents (~42 min)
start = time.time()
contents_tokenized = [bert_tokenizer.tokenize(item) for item in contents_short]
content_tokenization_time = time.time() - start
print('Tokenizing contents took ' + str(round(content_tokenization_time/60,2)) + ' min')

### writing tokenized contents list to csv file
with open("contents_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(contents_tokenized)



