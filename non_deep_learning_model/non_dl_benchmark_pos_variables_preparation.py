########## Non-DL Benchmark POS variables ##########
import csv
import time
import nltk
import numpy as np

# downloading nltk resources 
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')

##################################################
affix = 'allsides_duplicates_removed' # 'semeval'#
##################################################

non_dl_directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(non_dl_directory_path)

# loading tokenized articles
with open(f"non_dl_{affix}_content_tokenized.csv", newline='') as f:
    csvread = csv.reader(f)
    content_tokenized = list(csvread)


# Using dictionary to convert tags to ids to save memory 
tag_list = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN',
            'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$',
            'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB',
            'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
tag_dict = {tag: i for i,tag in enumerate(tag_list)}
reversed_tag_dict = {i: tag for tag,i in tag_dict.items()}
# nltk.help.upenn_tagset()

# convert articles into numpy arrays of ints corresponding to tag in tag_dict (75 min)
start = time.time()
contents_pos_tags = [np.array([tag_dict.get(pair[1]) for pair in nltk.pos_tag(article)]) 
                      for article in content_tokenized]
print(f'POS tags time: {(time.time() -start)//60} min')

del(content_tokenized)

### Creating normalized POS variables
start = time.time()
pos_variables_list = [np.array([np.sum(article_array==i)/len(article_array) for i in range(len(tag_dict))]).reshape(1,-1) 
                      for article_array in contents_pos_tags]
print('POS variable list time: ' + str(time.time() -start))

# create numpy array of shape (num_articles,num_pos_variables)
pos_variables = np.concatenate(pos_variables_list,axis=0)

# drop very rare cases (type appears in less than 1% of articles)
drop_rate = int(len(pos_variables)*0.01)

kept_ids = np.arange(len(tag_dict))[np.sum(pos_variables, axis=0)>drop_rate]
dropped_ids = np.arange(len(tag_dict))[np.sum(pos_variables, axis=0)<=drop_rate]

kept_pos = [reversed_tag_dict[i] for i in kept_ids]
dropped_pos = [reversed_tag_dict[i] for i in dropped_ids]

### choose same pos variables for semeval
if affix=='semeval':
    kept_pos = ['TO','VBN','VBG','JJ','VBZ','VBP','NN','DT','PRP','PRP$','RB','VBD','IN','MD','VB','NNS','CC','CD']
    kept_ids = np.array([tag_dict[label] for label in kept_pos])


pos_variables_used = pos_variables[:,kept_ids]

# kept: 'TO','VBN','VBG','JJ','VBZ','VBP','NN','DT','PRP','PRP$','RB','VBD','IN',
#       'MD','VB','NNS','CC','CD'


print(f' {len(dropped_pos)} variables dropped because of frequency below 1%, {len(kept_pos)} remaining ')
print(f'{dropped_pos} are dropped and {kept_pos} are kept')

np.save(f'non_dl_{affix}_pos_variables.npy', pos_variables_used)








