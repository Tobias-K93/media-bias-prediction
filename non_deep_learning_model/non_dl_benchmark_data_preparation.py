########### Non-DL Benchmark data preparation ##########
import time
import csv
import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

##################################################
affix = 'allsides_duplicates_removed' # 'semeval'#
##################################################

# path to media-bias-prediction repository
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# directory to save all variables to
os.chdir(os.path.join(repo_path,'non_deep_learning_model'))


if affix=='semeval'
    data = pd.read_csv(os.path.join(repo_path, 'sem_eval', 'semeval_data.tsv'),sep='\t', 
                           names=['hyperpartisan_label', 'unknown1', 'unknown2','content', 'title'])

    contents = list(data['content'])
    contents = [article.replace('<splt>', '') for article in contents]
    
    bias_array = np.array(data['hyperpartisan_label'])
    np.save(f'non_dl_semeval_bias.npy', bias_array) 

elif affix == 'allsides_duplicates_removed':
    data = pd.read_csv(os.path.join(repo_path, 'data_preparation','allsides_data', f'{affix}_data_short.csv'))

    contents = list(data['content'])

    bias_array = np.array(data['bias'])
    np.save(f'non_dl_{affix}_bias.npy', bias_array)

    source_array = np.array(data['source'])
    np.save(f'non_dl_{affix}_source.npy', source_array)

    del(data, bias_array, source_array)
else:
    raise AssertionError('affix should be \'semeval\' or \'allsides_duplicates_removed\'')



total_time_start = time.time()

### Counting characters
# Quotes
start = time.time()
quotes_count = np.reshape(np.array([len([1 for character in article if character=='"'])
                         for article in contents], ndmin=1),(-1,1))
print(f'Quotes time: {round(time.time()-start,2)} sec')

# Uppercase letters
start = time.time()
uppercase_count = np.reshape(np.array([len([1 for character in article if character.isupper()])
                            for article in contents], ndmin=1),(-1,1))
print(f'Uppercase time: {round(time.time()-start,2)} sec')

# total characters
start = time.time()
character_count = np.reshape(np.array([len(article) for article in contents]),(-1,1))
print(f'Total count time: {round(time.time()-start,2)} sec')

# Periods
start = time.time()
period_count = np.reshape(np.array([len([1 for character in article 
                          if character=='.']) for article in contents]),(-1,1))
print(f'Periods time: {round(time.time()-start,2)} sec')

# Question marks
start = time.time()
questionmark_count = np.reshape(np.array([len([1 for character in article 
                          if character=='?']) for article in contents]),(-1,1))
print(f'Question marks time: {round(time.time()-start,2)} sec')

# Exclamation marks
start = time.time()
exclamationmark_count = np.reshape(np.array([len([1 for character in article 
                          if character=='!']) for article in contents]),(-1,1))
print(f'Exclamation marks time: {round(time.time()-start,2)} sec')

# Digits
digit_count = np.reshape(np.array([len([1 for character in article 
                          if character.isdigit()]) for article in contents]),(-1,1))


character_variables = np.concatenate((quotes_count, uppercase_count, character_count, 
                                      period_count, questionmark_count, 
                                      exclamationmark_count, digit_count), axis=1)

np.save(f'non_dl_{affix}_character_variables.npy', character_variables)

del(quotes_count, uppercase_count, character_count, period_count, exclamationmark_count,
    questionmark_count, digit_count, character_variables)


### prepare contents for word based variables

# removing punctuation except for \\
start = time.time()
contents = [''.join([character for character in article if character not in '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~']) for article in contents]
print(f'Punctuation removement time: {round(time.time()-start,2)} sec')

# simple space splitting tokenization  
start = time.time()
content_tokenized = []
for article in contents:
    # splitting on space
    token_list = article.split(' ')
    # removing \r\n\r\n 
    new_token_list = []
    for token in token_list:
        new_token_list = new_token_list + token.split('\r\n\r\n')
    # removing empty entries '' and make words lower case
    new_token_list = [token.lower() for token in new_token_list if token!='']
    content_tokenized.append(new_token_list)

print(f'Tokenization time: {round(time.time()-start,2)} sec')


### writing tokenized content to csv for faster use later on
with open(f"non_dl_{affix}_content_tokenized.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(content_tokenized)

###### Average number of words per sentence
contents_sentences = [sent_tokenize(article) for article in contents]

average_sentence_length = np.array([sum([len(sentence.split(' ')) 
                                    for sentence in contents_sentences[i_article]])
                                    /len(content_tokenized[i_article]) 
                                    for i_article in range(len(contents_sentences))]).reshape((-1,1))
        

del(contents)

### Applying dictionaries, loading as empty dictionaries for faster querrying over keys 

# Recasens (2013) dictionaries
os.chdir(os.path.join(repo_path, 'non_deep_learning_model', 'recasens_2013_dicionaries'))

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

# Barron- Cedeno (2019) dictionaries
os.chdir(os.path.join(repo_path, 'non_deep_learning_model', 'barron_cedeno_2019_dictionaries'))

authority_dict = {}
with open("Authority_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        authority_dict[line[0]] = None

fairness_dict = {}
with open("Fairness_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        fairness_dict[line[0]] = None

harm_dict = {}
with open("Harm_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        harm_dict[line[0]] = None

ingroup_dict = {}
with open("Ingroup_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        ingroup_dict[line[0]] = None

purity_dict = {}
with open("Purity_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        purity_dict[line[0]] = None

negative_strong_subjective_dict = {}
with open("negative_strong_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        negative_strong_subjective_dict[line[0]] = None

negative_weak_subjective_dict = {}
with open("negative_weak_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        negative_weak_subjective_dict[line[0]] = None

neutral_strong_subjective_dict = {}
with open("neutral_strong_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        negative_strong_subjective_dict[line[0]] = None

neutral_weak_subjective_dict = {}
with open("neutral_weak_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        neutral_weak_subjective_dict[line[0]] = None

positive_strong_subjective_dict = {}
with open("positive_strong_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        positive_strong_subjective_dict[line[0]] = None

positive_weak_subjective_dict = {}
with open("positive_weak_subjective_dictionary.txt", newline='') as f:
    csvread = csv.reader(f,delimiter='\n')
    for line in csvread:
        positive_weak_subjective_dict[line[0]] = None

# Pronouns dictionaries

first_person_dict = {'i': None, 'my': None, 'me': None, 'myself': None}
second_person_dict = {'you': None, 'your': None, 'yourself': None}


os.chdir(os.path.join(repo_path,'non_deep_learning_model'))

# putting all dictionaries into list for looping
dictionaries_list = [assertives_dict, bias_lexicon_dict, factives_dict, hedges_dict, 
                     implicatives_dict, opinion_negative_dict, opinion_positive_dict, 
                     report_verbs_dict, authority_dict, fairness_dict, harm_dict, 
                     ingroup_dict, purity_dict, negative_strong_subjective_dict, 
                     negative_weak_subjective_dict, neutral_strong_subjective_dict, 
                     neutral_weak_subjective_dict, positive_strong_subjective_dict, 
                     positive_weak_subjective_dict, first_person_dict, 
                     second_person_dict]


dict_count_list = []
for i,dictionary in enumerate(dictionaries_list):
    start = time.time()
    dict_count_list.append(np.reshape(np.array([len([1 for token in article if token in dictionary]) 
                                                for article in content_tokenized]),(-1,1)))
    end = time.time() - start
    print(f'number {i+1} of {len(dictionaries_list)} dictionaries done, took {round(end)} sec' )


dict_count_variables = np.concatenate(dict_count_list, axis=1)

np.save(f'non_dl_{affix}_dict_count_variables.npy', dict_count_variables)

### Number of words per article
article_length = np.array([len(article) for article in content_tokenized]).reshape((-1,1))

### Average length of words
start = time.time()
average_word_length = np.reshape(np.array([sum([len(token) for token in article])/len(article) for article in content_tokenized]),(-1,1))
print(f'Avg. word length time: {round(time.time()-start,2)} sec')

##### Vocabulary richness features from Barron-Cedeno (2019)
type_token_ratio_list = []
hapax_legomena_list = []
hapax_dislegomena_list = []
honores_r_list = []
yules_k_list = []

start = time.time()
for article in content_tokenized:
    token_type_count_dict = {}
    for token in article:
        try:
            token_type_count_dict[token] += 1
        except KeyError:
            token_type_count_dict[token] = 1
    # Type to token ratio        
    type_token_ratio_list.append(len(token_type_count_dict)/len(article))
    # Hapax legomena: Number of types appearing only once
    hapax_legomena = np.sum(np.array(list(token_type_count_dict.values()))==1)
    hapax_legomena_list.append(hapax_legomena)
    # Hapax dislegomena: Types appearing only twice
    hapax_dislegomena_list.append(np.sum(np.array(list(token_type_count_dict.values()))==2))
    # Honore's R: (100*log(|tokens|)/(1-hapax_legomena/|types|))
    honores_r = (100*np.log(len(article)))/(1-hapax_legomena/len(token_type_count_dict))
    honores_r_list.append(honores_r)
    # Yule’s characteristic K: 10⁴*(sum(rank_i²*frequency_i)-total_tokens)/totel_tokens²
    rank, frequency = np.unique(np.array(list(token_type_count_dict.values())), return_counts=True)
    yules_k = 10000*((np.sum((rank**2)*frequency)-len(article))/len(article)**2)
    yules_k_list.append(yules_k)

type_token_ratio_array = np.array(type_token_ratio_list).reshape((-1,1))
hapax_legomena_array = np.array(hapax_legomena_list).reshape((-1,1))
hapax_dislegomena_array = np.array(hapax_dislegomena_list).reshape((-1,1))
honores_r_array = np.array(honores_r_list).reshape((-1,1))
yules_k_array = np.array(yules_k_list).reshape((-1,1))

print(f'Vocabulary richness features time: {round(time.time()-start,2)} sec')


token_based_variables = np.concatenate((average_sentence_length,
                                        article_length, 
                                        average_word_length,
                                        type_token_ratio_array,
                                        hapax_legomena_array,
                                        hapax_dislegomena_array,
                                        honores_r_array,
                                        yules_k_array),axis=1)

np.save(f'non_dl_{affix}_token_based_variables.npy', token_based_variables)

total_time_end = time.time() - total_time_start
total_time_min = int(total_time_end // 60)
total_time_sec = int(total_time_end % 60)

print(f'Total time: {total_time_min} min and {total_time_sec} sec') 

