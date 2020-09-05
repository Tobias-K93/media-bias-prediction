########## Non-DL Benchmark POS variables ##########
    ########## part of speach tagging ##########
import csv
import time
import nltk
import numpy as np

# downloading nltk resources 
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')

# loading tokenized articles
with open("non_dl_content_tokenized.csv", newline='') as f:
    csvread = csv.reader(f)
    content_tokenized = list(csvread)


# Using dictionary to convert tags to ids to save memory 
tag_list = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
tag_dict = {tag: i for i,tag in enumerate(tag_list)}
# nltk.help.upenn_tagset()

# convert articles into numpy arrays of ints corresponding to tag in tag_dict
start = time.time()
contents_pos_tags = [np.array([tag_dict.get(pair[1]) for pair in nltk.pos_tag(article)]) 
                      for article in content_tokenized]
print('POS nouns time: ' + str(time.time() -start))

del(content_tokenized)

### Creating POS variables

pos_variables_list = [np.array([np.sum(article_array==i) for i in range(len(tag_dict))]).reshape(1,-1) 
                      for article_array in contents_pos_tags]

# create numpy array of shape (num_articles,num_pos_variables)
pos_variables = np.concatenate(pos_variables_list,axis=0)

np.save('non_dl_pos_variables.npy', pos_variables)
































