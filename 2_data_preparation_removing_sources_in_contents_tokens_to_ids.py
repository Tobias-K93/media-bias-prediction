##### Removing sources in contents and converting tokens to ids #####
import time
import csv
import transformers

# load tokenizer class
tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

# create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

### reading tokenized files
with open("titles_tokenized.csv", newline='') as f:
    csvread = csv.reader(f)
    titles_tokenized = list(csvread)

with open("contents_tokenized.csv", newline='') as f:
    csvread = csv.reader(f)
    contents_tokenized = list(csvread)

with open("sources_tokenized.csv", newline='') as f:
    csvread = csv.reader(f)
    sources_tokenized = list(csvread)

### Removing sources from contents
# loop over text contents and coresponding news sources
for source,content in zip(sources_tokenized,contents_tokenized):
    # loop over each token in content and give coresponding id
    for i,token in enumerate(content):
        # if - elif clauses checking if (token corresponds with first token 
        # of source-tokens and is longer than 1)
        # if not: check if only token corresponds with source-token
        # in each case delete tokens, otherwise do nothing

        # length >1
        if (token == source[0]) and (len(source)>1) and (len(content)>len(content[:i+1])):
            # length >2
            if (len(source)>2) and (len(content)>len(content[:i+2])):
                # length >3
                if len(source)>3:
                    # length >4
                    if len(source)>4:
                        del(content[i:i+len(source)])
                    # length 4
                    elif content[i:i+4] == source[:4]:
                        del(content[i:i+4]) 
                # lenght 3
                elif content[i:i+3] == source[:3]:
                    del(content[i:i+3])                
            # length 2
            elif (content[i+1] == source[1]) and (len(content)>len(content[:i+1])):
                del(content[i:i+2])
        # length 1
        elif (token == source[0]) and (len(content)>len(content[:i+1])):
            del(content[i])

### deleting sources
try:
    del(sources_tokenized)
except NameError:
    pass

# convert tokens to ids
titles_ids = [bert_tokenizer.convert_tokens_to_ids(item) for item in titles_tokenized]
contents_ids = [bert_tokenizer.convert_tokens_to_ids(item) for item in contents_tokenized]

# write id files
with open("titles_ids.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(titles_ids)

with open("contents_ids_source_removed.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(contents_ids)

