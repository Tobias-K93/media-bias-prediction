########### Non-DL Benchmark data preparation ##########
import time
import csv
import os
import string
import pandas as pd
import numpy as np
import nltk

data = pd.read_csv('/home/tobias/Documents/Studium/Master_thesis/programming/data_short.csv')

contents = list(data['content'])
del(data)

### Counting characters
# Quotes
start = time.time()
quotes_count = np.reshape(np.array([len([1 for character in article if character=='"'])
                         for article in contents], ndmin=1),(-1,1))
print('Quotes time: ' + str(time.time()-start))

# Uppercase letters
start = time.time()
uppercase_count = np.reshape(np.array([len([1 for character in article if character.isupper()])
                            for article in contents], ndmin=1),(-1,1))
print('Uppercase time: ' + str(time.time()-start))

# total characters
start = time.time()
character_count = np.reshape(np.array([len(article) for article in contents]),(-1,1))
print('Total count time: ' + str(time.time()-start))

# Periods
start = time.time()
period_count = np.reshape(np.array([len([1 for character in article 
                          if character=='.']) for article in contents]),(-1,1))
print('Periods time: ' + str(time.time()-start))

# Question marks
start = time.time()
questionmark_count = np.reshape(np.array([len([1 for character in article 
                          if character=='?']) for article in contents]),(-1,1))
print('Question marks time: ' + str(time.time()-start))

# Exclamation marks
start = time.time()
exclamationmark_count = np.reshape(np.array([len([1 for character in article 
                          if character=='!']) for article in contents]),(-1,1))
print('Exclamation marks time: ' + str(time.time()-start))


character_variables = np.concatenate((quotes_count, uppercase_count, character_count, 
                                      period_count, questionmark_count, 
                                      exclamationmark_count), axis=1)

np.save('non_dl_character_variables.npy', character_variables)

del(quotes_count, uppercase_count, character_count, period_count, 
character_variables)


### prepare contents for word based variables

# removing punctuation except for \\
start = time.time()
contents = [''.join([character for character in article if character not in '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~']) for article in contents]
print('Punctuation time: ' + str(time.time() -start))

# "tokenize" 
start = time.time()
content_tokenized = []
for article in contents:
    # splitting on space
    token_list = article.split(' ')
    # removing \r\n\r\n 
    new_token_list = []
    for token in token_list:
        new_token_list = new_token_list + token.split('\r\n\r\n')
    # removing empty entries ''
    new_token_list = [token for token in new_token_list if token!='']
    content_tokenized.append(new_token_list)

print('Tokenization time: ' + str(time.time() -start))


### writing tokenized content to csv for faster use later on
with open("non_dl_content_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(content_tokenized)

del(contents)

### Applying dictionaries 
os.chdir('/home/tobias/Documents/Studium/Master_thesis/programming/non_dl_benchmark/bias-lexicon_Recasens(2013)')

# Loading as empty dictionaries for faster querrying over keys
assertives_dict = {}
with open("assertives_hooper1975.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        assertives_dict[line[0]] = None

bias_lexicon_dict = {}
with open("bias-lexicon.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        bias_lexicon_dict[line[0]] = None

factives_dict = {}
with open("factives_hooper1975.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        factives_dict[line[0]] = None

hedges_dict = {}
with open("hedges_hyland2005.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        hedges_dict[line[0]] = None

implicatives_dict = {}
with open("implicatives_karttunen1971.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        implicatives_dict[line[0]] = None

opinion_negative_dict = {}
with open("opinion_lexicon_negative-words.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        opinion_negative_dict[line[0]] = None

opinion_positive_dict = {}
with open("opinion_lexicon_positive-words.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        opinion_positive_dict[line[0]] = None

report_verbs_dict = {}
with open("report_verbs.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        report_verbs_dict[line[0]] = None

os.chdir('/home/tobias/Documents/Studium/Master_thesis/programming/non_dl_benchmark')

# putting all dictionaries into list for looping
dictionaries_list = [assertives_dict, bias_lexicon_dict, factives_dict, hedges_dict, implicatives_dict, 
                     opinion_negative_dict, opinion_positive_dict, report_verbs_dict]

dict_count_list = []
for i,dictionary in enumerate(dictionaries_list):
    start = time.time()
    dict_count_list.append(np.reshape(np.array([len([1 for token in article if token in dictionary]) 
                                                for article in content_tokenized]),(-1,1)))
    end = time.time() - start
    print('number ' + str(i+1) + ' of ' + str(len(dictionaries_list)) + 
          ' done, took ' + str(round(end)) + ' sec' )


dict_count_variables = np.concatenate(dict_count_list, axis=1)

np.save('non_dl_dict_count_variables.npy', dict_count_variables)

### Average length of words
start = time.time()
average_word_length = np.reshape(np.array([sum([len(token) for token in article])/len(article) for article in content_tokenized]),(-1,1))
print('Avg. word length time: ' + str(time.time() -start))

### Type/Token ratio
start = time.time()
type_token_ratio = np.array([len({token: None for token in article})/len(article) 
                             for article in content_tokenized]).reshape((-1,1))
print('Type/Token ratio time: ' + str(time.time() -start))

token_based_variables = np.concatenate((average_word_length,type_token_ratio),axis=1)

np.save('non_dl_token_based_variables.npy', token_based_variables)

