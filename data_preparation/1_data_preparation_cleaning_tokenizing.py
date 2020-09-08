########## data preparation for news bias dataset ##########
import time
import csv
import re
import os 
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch
from nltk.tokenize import sent_tokenize

#######################################
removing_duplicate_sentences = True  #
#######################################
affix = 'allsides' ###
#########################################################################
# bias_dict = {'extreme_left': 0, 'left_bias': 1, 'left_center_bias': 2,  #
#              'least_biased': 3, 'right_center_bias': 4, 'right_bias': 5,#
#              'extreme_right': 6}                                        #
#########################################################################
os.chdir('/home/tobias/Documents/Studium/Master_thesis/programming/allsides')

# load data
data = pd.read_csv('allsides_articles.csv')

# create lists of titles and contents
titles = list(data['name'])
contents = list(data['content'])

# ID's of entries without title (3)
titles_nans_id = [i for i,item in enumerate(titles) if pd.isna(item)]

# ID's of entries without content (218)
contents_nans_id = [i for i,item in enumerate(contents) if pd.isna(item)]
# Sources of articles without content and the corresponding frequencies
np.unique(data['source'].iloc[contents_nans_id], return_counts=True)

# removing entries with nans 
nans_id = list(set(titles_nans_id+contents_nans_id))
data.drop(index=nans_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

# reset titles and contents lists
del(titles,contents)
titles = list(data['name'])
contents = list(data['content'])

# get lengths of titles and contents 
contents_length = [len(item) for item in contents]

#########################################################################
# ### some stats on length in characters
# titles_length = [len(item) for item in titles]

# titles_length_max = max(titles_length)
# titles_length_quantiles = np.quantile(titles_length,[0.25,0.5,0.75])
# # histogram of lengths of titles
# plt.hist(titles_length)

# contents_length_max = max(contents_length)
# contents_length_quantiles = np.quantile(contents_length,[0.25,0.5,0.75])

##### Closer look at articles around the cutoff
# List of articles with wanted length

# contents_length_array = np.array(contents_length)
# list(np.arange(len(contents_length_array))[(contents_length_array > 900)*(contents_length_array < 1300)])

# It seems that most articles with little more than 500 characters
# are not real articles but either short summaries or short 
# snippets of longer articles, videos, and podcasts. It also 
# happens that they list topics of the the webside that are 
# discribed with one sentence each. 

# print(data['source'].iloc[104])
# print(contents[104])

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
#########################################################################
#########################
# Among the very short articles is a lot of noise (non-news content, 
# short description linking to full articles, error messages,...). Thus, removing articles
# which are uncommenly short (under 500 characters ~ 100 words) should 
# reduce noise. Similarly very long articles (over 20000 characters ~ 4000 
# words) contain unwanted and rather rare cases e.g. several whole articles, 
# that belong to a specific topic, listed as one. 

# 36910 short cut
short_cut = np.sum(np.array(contents_length)<500)
# 1391 long cut
long_cut = np.sum(np.array(contents_length)>20000)

contents_unwanted_length_id = [i for i,item in enumerate(contents_length) if item < 500 or item > 20000]

try:
    del(contents_length)
except NameError:
    pass

# removing short and long articles from dataset
data.drop(index=contents_unwanted_length_id,inplace=True)

# reset index
data.reset_index(drop=True, inplace=True)

# ###### create target labels vector #####
# bias_list = [bias_dict[item] for item in data['bias']]
        
# bias_tensor = torch.tensor(bias_list)

# torch.save(bias_tensor,f'{affix}_bias_tensor.pt')

# # creating counting bias array for full mbfc
# bias_array = np.array(data['bias'])
# np.save('mbfc_full_for_counting_bias_array.npy', bias_array)

# # removing saved bias objects
# try:
#     del(bias_list,bias_tensor)
# except NameError:
#     pass
# ########################################

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
# due to information leakage (source: 'bearing arms', url: 'https://bearingarms.com/author/davidl/')
# or due to only containing noise ('https://t.co/BejuGwOJVD')
# about 4% of main dataset contains urls
# taken from https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

contents = [re.sub(r'http\S+', '', item) for item in contents]

data['name'] = titles
data['content'] = contents

##### Saving shortened dataset 
# data.to_csv(f'{affix}_short.csv', index=False)

### Remove repeating sentences per source
# sentences - articles - cut
# Newswars 35413 - 2738 - 5
# The Sun 664540 - 42296 - 50
# BBC 338046 - 13818 - 20
# Daily Kos 34633 - 924 - 4
# CNN 183509 - 8143 - 15
# Reuters 55866 - 3670 -10
# The Washington Examiner 7997 - 466 - 4
# The Daily Mirror 254058 - 12763 - 15
# Drudge Report 300302 - 17123 - 25

##### Finding examples for sentences #####
# source_articles = data['content'][data['source']=='Investors Business Daily']
 
# article_sentence_dict = {}
# for article in source_articles:
#     article_sentences_list = sent_tokenize(article)    
#     for sentence in article_sentences_list:
#         if sentence in article_sentence_dict:
#             article_sentence_dict[sentence] += 1
#         else:
#             article_sentence_dict[sentence] = 1

# duplicate_sentence_count_cut = int(min(max(4,len(source_articles)/500),50))

# article_sentence_dict_selected = {key: value for key, value in article_sentence_dict.items()
#                                   if value>50 and len(key)>12}

# print(len(source_articles))
# article_sentence_dict_selected
##########################################

if removing_duplicate_sentences:
    affix += '_duplicates_removed'

    data['content'] = contents
    del(contents)
    
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
        article_sentence_dict_selected = {key: value for key, value in article_sentence_dict.items()
                                        if value>duplicate_sentence_count_cut and len(key)>12}

        ### removing duplicate sentences 
        final_source_articles = []
        for article in source_articles:
            article_sentences_list = sent_tokenize(article)

            final_article = ''.join([sentence + ' ' for sentence in article_sentences_list if sentence not in article_sentence_dict_selected])
            final_source_articles.append(final_article[:-1])

        ### change articles to articles without duplicate sentences 
        # use .loc to make sure original dataframe is modified
        data.loc[data['source']==news_source,'content'] = final_source_articles
        # same as: data['content'].loc[data['source']=='Newswars'] = final_source_articles
    
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
#data['content'] = contents
data.drop(index=nans_id_after_duplicate_removing,inplace=True)

### saving shortened dataset of which duplicates were removed
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

########################################
### Tokenize sources for later removal from content
# sources = list(data['source'])

# start = time.time()
# sources_tokenized = [bert_tokenizer.tokenize(item) for item in sources]
# source_tokenization_time = time.time() - start
# print('Tokenizing sources took ' + str(source_tokenization_time) + ' sec')

# ### writing tokenized sources list to csv file
# with open("sources_tokenized.csv","w", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(sources_tokenized)

# ### delete saved sources objects
# try:
#     del(sources, sources_tokenized)
# except NameError:
#     pass

########################################

try:
    del(data)
except NameError:
    pass


### tokenize titles
start = time.time()
titles_tokenized = [bert_tokenizer.tokenize(item) for item in titles]
title_tokenization_time = time.time() - start
print('Tokenizing titles took ' + str(title_tokenization_time) + ' sec')

### writing tokenized titles list to csv file
with open("titles_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(titles_tokenized)

# ### delete saved titles objects
# try:
#     del(titles, titles_tokenized)
# except NameError:
#     pass

# # tokenize contents (~42 min to 72 min)
# start = time.time()
# contents_tokenized = [bert_tokenizer.tokenize(item) for item in contents_short]
# content_tokenization_time = (time.time() - start)/60
# print(f'Tokenizing contents took {content_tokenization_time:.2} min')

# ### writing tokenized contents list to csv file
# with open(f"{affix}contents_tokenized.csv","w", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(contents_tokenized)



