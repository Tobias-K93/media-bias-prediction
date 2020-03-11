#################### Creating labels table #################### 

import pandas as pd 
import numpy as np
# import os
# os.chdir('/home/tobias/Documents/Studium/Master_thesis')

labels = pd.read_csv('/home/tobias/Documents/Studium/Master_thesis/media_bias/labels.csv')

labels.rename(columns={'Unnamed: 0': 'Source'}, inplace=True)

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
 'Open Sources, state','BuzzFeed, leaning',
 'PolitiFact, Pants on Fire!',
 'PolitiFact, False',
 'PolitiFact, Mostly False',
 'PolitiFact, Half-True',
 'PolitiFact, Mostly True',
 'PolitiFact, True']

labels_wanted = labels.drop(unwanted_columns, axis=1)


# overview Media Bias/Fact Check
print(np.unique(labels_wanted['Media Bias / Fact Check, label'].dropna(), return_counts=True))
# labeled: least biased
mbfc_least_biased = list(labels_wanted['Source'][labels_wanted['Media Bias / Fact Check, label']=='least_biased'])
# labeled: left bias
mbfc_left_bias = list(labels_wanted['Source'][labels_wanted['Media Bias / Fact Check, label']=='left_bias'])
# labeled: right bias
mbfc_right_bias = list(labels_wanted['Source'][labels_wanted['Media Bias / Fact Check, label']=='right_bias'])
# variable: extreme left
mbfc_extreme_left = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_left']==1]['Source'])
# variable: extreme right
mbfc_extreme_right = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
                    ['Media Bias / Fact Check, extreme_right']==1]['Source'])
# variable: right (not obvious why there is an variable: right; only contains one outlet)
mbfc_right = list(labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
            [labels_wanted.dropna(subset=['Media Bias / Fact Check, right'])
            ['Media Bias / Fact Check, right']==1]['Source'])


### overview Allsides
print(np.unique(labels_wanted['Allsides, bias_rating'].dropna(), return_counts=True))
# labeled: center
allsides_center = list(labels_wanted['Source'][labels_wanted['Allsides, bias_rating']=='Center'])
# labled: right
allsides_right = list(labels_wanted['Source'][labels_wanted['Allsides, bias_rating']=='Right'])
# labled: left
allsides_left = list(labels_wanted['Source'][labels_wanted['Allsides, bias_rating']=='Left'])


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

# sql_string = ''
# for source in outlets:
#     sql_string +='NOT source=' + "'" + source + "'"+ ' AND '

# sql_string = ''
# for source in right_checked:
#     sql_string += 'source=' + "'" + source + "'" + ' OR '

# sql_string = ''
# for source in center_checked:
#     sql_string += 'source=' + "'" + source + "'" + ' OR '
# print(sql_string)
