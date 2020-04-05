###### Data Cleaning Function
import time
import csv
import re
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch

def data_preparation(data, name_string, bias_dict={'moderate_left': 0, 'moderate_right': 1}):
    """
    data: pandas DataFrame 
    name_string: phrase added to saved files (add _ at the end of the phrase)
    bias_dict: dictionary that maps bias label from string to integer
    """

    # create lists of titles and contents
    titles = list(data['name'])
    contents = list(data['content'])

    # ID's of entries without title
    titles_nans_id = [i for i,item in enumerate(titles) if pd.isna(item)]

    # ID's of entries without content 
    contents_nans_id = [i for i,item in enumerate(contents) if pd.isna(item)]

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

    # number short cut
    short_cut = np.sum(np.array(contents_length)<500)
    print(f'{short_cut} articles dropped due to short length')
    # number long cut
    long_cut = np.sum(np.array(contents_length)>20000)
    print(f'{long_cut} articles dropped due to long length')
    
    contents_unwanted_length_id = [i for i,item in enumerate(contents_length) if item < 500 or item > 20000]

    # removing short and long articles from dataset
    data.drop(index=contents_unwanted_length_id,inplace=True)

    # reset index
    data.reset_index(drop=True, inplace=True)

    ##### Saving shortened dataset 
    data.to_csv(f'{name_string}data_short.csv', index=False)

    ############# create target labels vector #############

    bias_list = [bias_dict[item] for item in data['bias']]
            
    bias_tensor = torch.tensor(bias_list)

    torch.save(bias_tensor,f'{name_string}bias_tensor.pt')
    #######################################################
    # reset titles and contents lists
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


    # Shortening contents to max 5000 characters (for faster tokenization)
    contents_short = [item[:5000] for item in contents]

    # delete dataframe, bias objects, and long contents
    try:
        del(data, bias_list, bias_tensor, contents)
    except NameError:
        pass

    # load tokenizer class
    tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')

    # create bert_tokenizer object from pretrained weights, i.e. loading vocabulary
    bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    ### tokenize titles
    start = time.time()
    titles_tokenized = [bert_tokenizer.tokenize(item) for item in titles]
    title_tokenization_time = time.time() - start
    print('Tokenizing titles took ' + str(title_tokenization_time) + ' sec')

    ### tokenize contents (~42 min/ ~38 min)
    start = time.time()
    contents_tokenized = [bert_tokenizer.tokenize(item) for item in contents_short]
    content_tokenization_time = time.time() - start
    print('Tokenizing contents took ' + str(round(content_tokenization_time/60,2)) + ' min')

    ### converting tokens to ids, truncate, pad, and create masks
    # for titles
    truncation_value =  50 # like in main dataset
    titles_text = []
    titles_mask = []
    
    for item in titles_tokenized:
        output_dict = bert_tokenizer.encode_plus(item, 
                                                 max_length= truncation_value,
                                                 pad_to_max_length=True,
                                                #  return_tensors='pt',
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

    ### save tensors
    torch.save(titles_text_tensor,f'{name_string}titles_text_tensor.pt')
    torch.save(titles_mask_tensor,f'{name_string}titles_mask_tensor.pt')

    torch.save(contents_text_tensor,f'{name_string}contents_text_tensor.pt')
    torch.save(contents_mask_tensor,f'{name_string}contents_mask_tensor.pt')





