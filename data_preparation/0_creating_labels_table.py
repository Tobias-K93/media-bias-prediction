#################### Creating labels table #################### 

import pandas as pd 
import numpy as np
# import os
# os.chdir('/home/tobias/Documents/Studium/Master_thesis')

labels = pd.read_csv('/home/tobias/Documents/Studium/Master_thesis/media_bias_literature/labels.csv')

labels.rename(columns={'Unnamed: 0': 'Source'}, inplace=True)

# Removing German articles of Spiegel
labels.drop(index=np.array(labels.index)[labels['Source']=='Spiegel'], inplace=True)
labels.reset_index(drop=True, inplace=True)


unwanted_columns = ['NewsGuard, Does not repeatedly publish false content',
 'NewsGuard, Gathers and presents information responsibly',
 'NewsGuard, Regularly corrects or clarifies errors',
 'NewsGuard, Handles the difference between news and opinion responsibly',
 'NewsGuard, Avoids deceptive headlines',
 'NewsGuard, Website discloses ownership and financing',
 'NewsGuard, Clearly labels advertising',
 "NewsGuard, Reveals who's in charge, including any possible conflicts of interest",
 'NewsGuard, Provides information about content creators',
 'NewsGuard, score',
 'NewsGuard, overall_class',
 'Pew Research Center, known_by_40%',
 'Pew Research Center, total',
 'Pew Research Center, consistently_liberal',
 'Pew Research Center, mostly_liberal',
 'Pew Research Center, mixed',
 'Pew Research Center, mostly conservative',
 'Pew Research Center, consistently conservative',
 'Wikipedia, is_fake',
 'Open Sources, reliable',
 'Open Sources, fake',
 'Open Sources, unreliable',
 'Open Sources, bias',
 'Open Sources, conspiracy',
 'Open Sources, hate',
 'Open Sources, junksci',
 'Open Sources, rumor',
 'Open Sources, blog',
 'Open Sources, clickbait',
 'Open Sources, political',
 'Open Sources, satire',
 'Open Sources, state',
 'PolitiFact, Pants on Fire!',
 'PolitiFact, False',
 'PolitiFact, Mostly False',
 'PolitiFact, Half-True',
 'PolitiFact, Mostly True',
 'PolitiFact, True']
# 'BuzzFeed, leaning'
labels_wanted = labels.drop(unwanted_columns, axis=1)

##### Main Dataset: ##################################################
### overview Media Bias/Fact Check
print(np.unique(labels_wanted['Media Bias / Fact Check, label'].dropna(), 
                return_counts=True))
# labeled: least biased
mbfc_least_biased = list(labels_wanted['Source']
                         [labels_wanted['Media Bias / Fact Check, label']
                         =='least_biased'])
# labeled: left bias
mbfc_left_bias = list(labels_wanted['Source']
                [labels_wanted['Media Bias / Fact Check, label']
                =='left_bias'])
# labeled: right bias
mbfc_right_bias = list(labels_wanted['Source']
                 [labels_wanted['Media Bias / Fact Check, label']
                 =='right_bias'])
# variable: extreme left
mbfc_extreme_left = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_left']==1]['Source'])
# variable: extreme right
mbfc_extreme_right = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_right']==1]['Source'])
# variable: right (not obvious why there is a variable: right; only contains one outlet)
mbfc_right = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
            [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
            ['Media Bias / Fact Check, right']==1]['Source'])

### overview Allsides
print(np.unique(labels_wanted['Allsides, bias_rating'].dropna(), return_counts=True))

# labeled: center
allsides_center = list(labels_wanted['Source']
                      [labels_wanted['Allsides, bias_rating']=='Center'])
# labeled: right
allsides_right = list(labels_wanted['Source']
                     [labels_wanted['Allsides, bias_rating']=='Right'])
# labeled: left
allsides_left = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Left'])

# combining all
right = list(set(mbfc_right_bias + mbfc_extreme_right + mbfc_right + allsides_right))
left = list(set(mbfc_extreme_left + mbfc_left_bias + allsides_left))
center = list(set(mbfc_least_biased+ allsides_center))

dropped_sources = []

right_checked = []
for source in right:
    if source in left+center:
        dropped_sources.append(source)
    else:
        right_checked.append(source)

left_checked = []
for source in left:
    if source in right+center:
        if source not in dropped_sources:
            dropped_sources.append(source)
    else:
        left_checked.append(source)

center_checked = []
for source in center:
    if source in left+right:
        if source not in dropped_sources:
            dropped_sources.append(source)
    else:
        center_checked.append(source)


outlets = right_checked + left_checked + center_checked
bias_labels = len(right_checked)*['right'] + len(left_checked)*['left'] + len(center_checked)*['center']

# saving to csv
wanted_sources_with_labels = pd.DataFrame({'Source': outlets,'bias':bias_labels})
wanted_sources_with_labels.to_csv('bias_labels.csv', index=False)



##### Moderate Dataset ##################################################
### Media Bias / Fact Check
mbfc_left_center_bias = list(labels_wanted['Source']
                            [labels_wanted['Media Bias / Fact Check, label']
                            =='left_center_bias'])

mbfc_right_center_bias = list(labels_wanted['Source']
                        [labels_wanted['Media Bias / Fact Check, label']
                        =='right_center_bias'])
### Allsides
# labeled lean left
allsides_lean_left = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Lean Left'])
# labeled lean right
allsides_lean_right = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Lean Right'])

# combining all
moderate_right = list(set(mbfc_right_center_bias + allsides_lean_right))
moderate_left = list(set(mbfc_left_center_bias + allsides_lean_left))

dropped_moderate_sources = []

moderate_right_checked = []
for source in moderate_right:
    if source in left+center+right+moderate_left:
        dropped_moderate_sources.append(source)
    else:
        moderate_right_checked.append(source)

moderate_left_checked = []
for source in moderate_left:
    if source in right+center+left+moderate_right:
        if source not in dropped_moderate_sources:
            dropped_moderate_sources.append(source)
    else:
        moderate_left_checked.append(source)

moderate_outlets = moderate_right_checked + moderate_left_checked
moderate_bias_labels = len(moderate_right_checked)*['moderate_right'] \
                     + len(moderate_left_checked)*['moderate_left'] 

# saving to csv
moderate_sources_with_labels = pd.DataFrame({'Source': moderate_outlets,
                                             'bias':moderate_bias_labels})

moderate_sources_with_labels.to_csv('moderate_bias_labels.csv', index=False)

##### Allsides Dataset ##################################################

# labeled lean left
allsides_lean_left = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Lean Left'])
# labeled lean right
allsides_lean_right = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Lean Right'])
# labeled: center
allsides_center = list(labels_wanted['Source']
                      [labels_wanted['Allsides, bias_rating']=='Center'])
# labeled: right
allsides_right = list(labels_wanted['Source']
                     [labels_wanted['Allsides, bias_rating']=='Right'])
# labeled: left
allsides_left = list(labels_wanted['Source']
                    [labels_wanted['Allsides, bias_rating']=='Left'])

allsides_sources = allsides_center + allsides_lean_left + allsides_lean_right \
                   + allsides_left + allsides_right
allsides_bias_labels = len(allsides_center) * ['Center'] \
                     + len(allsides_lean_left) * ['Lean Left'] \
                     + len(allsides_lean_right) * ['Lean Right'] \
                     + len(allsides_left) * ['Left'] \
                     + len(allsides_right) * ['Right']

# saving to csv
allsides_sources_with_labels = pd.DataFrame({'Source': allsides_sources,
                                             'bias':allsides_bias_labels})


allsides_sources_with_labels.to_csv('allsides_bias_labels.csv', index=False)

##### TODO:
##### Media Bias / Fact Check dataset #############################################

# labeled: least biased
mbfc_least_biased = list(labels_wanted['Source']
                         [labels_wanted['Media Bias / Fact Check, label']
                         =='least_biased'])
# labeled: left bias
mbfc_left_bias = list(labels_wanted['Source']
                [labels_wanted['Media Bias / Fact Check, label']
                =='left_bias'])
# labeled: right bias
mbfc_right_bias = list(labels_wanted['Source']
                 [labels_wanted['Media Bias / Fact Check, label']
                 =='right_bias'])
# labeled: left center bias
mbfc_left_center_bias = list(labels_wanted['Source']
                            [labels_wanted['Media Bias / Fact Check, label']
                            =='left_center_bias'])
# labeled: right center bias
mbfc_right_center_bias = list(labels_wanted['Source']
                        [labels_wanted['Media Bias / Fact Check, label']
                        =='right_center_bias'])
# variable: extreme left
mbfc_extreme_left = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_left']==1]['Source'])
# variable: extreme right
mbfc_extreme_right = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_right']==1]['Source'])

###### Allsides out of sample testing dataset ##################################################

buzzfeed_left = list(labels_wanted['Source'][labels_wanted['BuzzFeed, leaning']=='left'])
buzzfeed_right = list(labels_wanted['Source'][labels_wanted['BuzzFeed, leaning']=='right'])

all_mbfc = (mbfc_least_biased + mbfc_extreme_left + mbfc_extreme_right
                            + mbfc_left_bias + mbfc_left_center_bias + mbfc_right_bias
                            + mbfc_right_center_bias)

all_mbfc_bias_labels = (len(mbfc_least_biased) * ['least_biased'] 
                        + len(mbfc_extreme_left) * ['extreme_left'] 
                        + len(mbfc_extreme_right) * ['extreme_right']
                        + len(mbfc_left_bias) * ['left_bias'] 
                        + len(mbfc_left_center_bias) * ['left_center_bias'] 
                        + len(mbfc_right_bias) * ['right_bias']
                        + len(mbfc_right_center_bias) * ['right_center_bias'])


all_mbfc_not_in_allsides = [source for source in all_mbfc if source not in allsides_sources]

all_mbfc_not_in_allsides_sources = []
all_mbfc_not_in_allsides_labels = []
for source,label in zip(all_mbfc,all_mbfc_bias_labels):
    if source not in allsides_sources:
        all_mbfc_not_in_allsides_sources.append(source)
        all_mbfc_not_in_allsides_labels.append(label)

all_mbfc_not_in_allsides_sources_with_labels = pd.DataFrame({'Source': all_mbfc_not_in_allsides_sources, 'bias': all_mbfc_not_in_allsides_labels})

all_mbfc_not_in_allsides_sources_with_labels.to_csv('media_bias_fact_check_wo_allsides/mbfc_bias_labels.csv', index=False)

# all_buzzfeed = buzzfeed_left + buzzfeed_right 
# all_buzzfeed_not_in_allsides_mbfc = [source for source in all_buzzfeed if source not in allsides_sources+all_mbfc]

#[source for source in all_buzzfeed_not_in_allsides_mbfc if source in buzzfeed_right]


##### SQL commands ##################################################
# sql_string = 'DELETE FROM articles WHERE '
# for source in all_mbfc_not_in_allsides:
#     sql_string +='NOT source=' + "'" + source + "'"+ ' AND '
# print(sql_string)

# sql_string = ''
# for source in right_checked:
#     sql_string += 'source=' + "'" + source + "'" + ' OR '

# sql_string = ''
# for source in center_checked:
#     sql_string += 'source=' + "'" + source + "'" + ' OR '
# print(sql_string)
