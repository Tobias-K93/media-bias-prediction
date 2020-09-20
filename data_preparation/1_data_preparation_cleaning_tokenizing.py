########## Data preparation - cleaning and tokenizing ##########
import time
import csv
import re
import os 
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import transformers
import torch
from nltk.tokenize import sent_tokenize

### Choose whether frequent sentences should be removed 
#######################################################
removing_duplicate_sentences = False  ###
########################################

### Select name of dataset to name files
########################################
affix = 'allsides' # 'mbfc' ###
###############################
# (mbfc only used for dataset comparison statistics, 
#  not used for model training)

# Respective dictionaries to convert bias labels from strings to ints
if affix == 'allsides':
    bias_dict = {'Left': 0, 'Lean Left': 1, 'Center': 2, 'Lean Right': 3, 'Right': 4}
elif affix == 'mbfc':
    bias_dict = {'extreme_left': 0, 'left_bias': 1, 'left_center_bias': 2, 
                'least_biased': 3, 'right_center_bias': 4, 'right_bias': 5,
                'extreme_right': 6}
else:
    raise AssertionError('affix should be \'allsides\' or \'mbfc\'')

# set path to data directory and make it working directory
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f'{affix}_data')
os.chdir(data_path)

# load data
data = pd.read_csv(f'{affix}_articles.csv')

# add bias labels to data according to source
bias_source_matching = pd.read_csv(f'{affix}_bias_labels.csv', header=0, names=['source', 'bias'])

data = data.merge(bias_source_matching)

# create lists of titles and contents
titles = list(data['name'])
contents = list(data['content'])

# ID's of entries without title (3)
titles_nans_id = [i for i,item in enumerate(titles) if pd.isna(item)]

# ID's of entries without content (218)
contents_nans_id = [i for i,item in enumerate(contents) if pd.isna(item)]

# removing entries with nans 
nans_id = list(set(titles_nans_id+contents_nans_id))
data.drop(index=nans_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

# reset titles and contents lists
del(titles,contents)
titles = list(data['name'])
contents = list(data['content'])

# get lengths of contents 
contents_length = [len(item) for item in contents]

# short cut (36910)
short_cut = np.sum(np.array(contents_length)<500)
# long cut (1391)
long_cut = np.sum(np.array(contents_length)>20000)

### remove extremely short and long articles to reduce noise
contents_unwanted_length_id = [i for i,item in enumerate(contents_length) if item < 500 or item > 20000]

try:
    del(contents_length)
except NameError:
    pass

# removing short and long articles from dataset
data.drop(index=contents_unwanted_length_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

###### create target labels vector #####
bias_list = [bias_dict[item] for item in data['bias']]
        
bias_tensor = torch.tensor(bias_list)

torch.save(bias_tensor,f'{affix}_bias_tensor.pt')

# removing saved bias objects
try:
    del(bias_list,bias_tensor)
except NameError:
    pass
########################################

# reset titles and contents lists
try:
    del(titles,contents)
except: 
    pass

titles = list(data['name'])
contents = list(data['content'])

### adjust html coding to unicode in a few articles (e.g. "&#8217" to "'")
html_parser = HTMLParser()

titles = [html_parser.unescape(item) for item in titles]
contents = [html_parser.unescape(item) for item in contents]

### remove urls from contents with regular expression
# due to information leakage (e.g. source: 'bearing arms', url: 'https://bearingarms.com/author/davidl/')
# or due to only containing noise (e.g. 'https://t.co/BejuGwOJVD')
# about 4% of main dataset contains urls

# taken from https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
contents = [re.sub(r'http\S+', '', item) for item in contents]

data['name'] = titles
data['content'] = contents

del(contents)

##### Removing frequent sentences from data
if removing_duplicate_sentences:
    affix += '_duplicates_removed'

    start = time.time()
    unique_sources =  np.unique(data['source'])
    for news_source in unique_sources:
        source_articles = data['content'][data['source']==news_source]

        ### Finding duplicate sentences 
        article_sentence_dict = {}
        for article in source_articles:
            article_sentences_list = sent_tokenize(article)    
            for sentence in article_sentences_list:
                if sentence in article_sentence_dict:
                    article_sentence_dict[sentence] += 1
                else:
                    article_sentence_dict[sentence] = 1

        # choose only those that appear often enough in relation to article corpus 
        # size and are of a certain length. This assures that common phrases 
        # (e.g. "No." "Yes." "Why?" "he added.") are not deleted
        # cut between 4 and 50 repetitions
        duplicate_sentence_count_cut = int(min(max(4,len(source_articles)/500),50))
        min_sentence_length = 12
        article_sentence_dict_selected = {key: value for key, value in article_sentence_dict.items()
                                        if value>duplicate_sentence_count_cut and len(key)>min_sentence_length}

        ### removing duplicate sentences 
        final_source_articles = []
        for article in source_articles:
            article_sentences_list = sent_tokenize(article)

            final_article = ''.join([sentence + ' ' for sentence in article_sentences_list if sentence not in article_sentence_dict_selected])
            final_source_articles.append(final_article[:-1])

        ### change articles to articles without duplicate sentences 
        data.loc[data['source']==news_source,'content'] = final_source_articles
    
    contents = list(data['content'])
    removing_duplicate_sentences_time = (time.time() - start)/60
    print(f'Removing duplicate sentences took {removing_duplicate_sentences_time:.2} min')

    try:
        del(article_sentence_dict, article_sentence_dict_selected)
    except NameError:
        pass

    # ID's of entries without content (due to repetitive content that was removed)
    nans_id_after_duplicate_removing = [i for i,item in enumerate(contents) if item=='']
    # removing entries with nans 
    data.drop(index=nans_id_after_duplicate_removing,inplace=True)

### saving shortened dataset 
data.to_csv(f'{affix}_data_short.csv', index=False)


### Shortening contents to max 4000 characters (for faster tokenization)
contents_short = [item[:4000] for item in contents]
# delete long contents
try:
    del(contents)
except NameError:
    pass

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

try:
    del(data)
except NameError:
    pass

### tokenize titles (not used for thesis)
# start = time.time()
# titles_tokenized = [bert_tokenizer.tokenize(item) for item in titles]
# title_tokenization_time = time.time() - start
# print('Tokenizing titles took ' + str(title_tokenization_time) + ' sec')

### writing tokenized titles list to csv file
# with open("titles_tokenized.csv","w", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(titles_tokenized)

# ### delete saved titles objects
# try:
#     del(titles, titles_tokenized)
# except NameError:
#     pass

# tokenize contents (~42 min to 72 min)
start = time.time()
contents_tokenized = [bert_tokenizer.tokenize(item) for item in contents_short]
content_tokenization_time = (time.time() - start)/60
print(f'Tokenizing contents took {content_tokenization_time:.2} min')

### writing tokenized contents list to csv file
with open(f"{affix}contents_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(contents_tokenized)



