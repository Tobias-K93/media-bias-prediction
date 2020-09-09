########## Data preparation - converting tokens to ids ########## 
import time
import csv
import transformers
import os

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

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

### reading tokenized files
# contents
with open(f"{affix}_contents_tokenized.csv", newline='') as f: 
    csvread = csv.reader(f)
    contents_tokenized = list(csvread)

# titles (not used for thesis)
# with open("titles_tokenized.csv", newline='') as f:
#     csvread = csv.reader(f)
#     titles_tokenized = list(csvread)


# convert tokens to ids
titles_ids = [bert_tokenizer.convert_tokens_to_ids(item) for item in titles_tokenized]
# contents_ids = [bert_tokenizer.convert_tokens_to_ids(item) for item in contents_tokenized]

### write id files
# contents
with open(f"{affix}_contents_ids.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(contents_ids)

# titles
# with open(f"{affix}_titles_ids.csv","w", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(titles_ids)

