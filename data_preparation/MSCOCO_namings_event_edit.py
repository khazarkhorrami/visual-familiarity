import os
import json
import scipy
from utilsMSCOCO import read_data_from_path, get_all_image_ids, get_all_cats
import cv2
from matplotlib import pyplot as plt
import nltk
import numpy as np
import copy



save_path =   '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/new/'  

#%%
######################### reading train and val data

dataDir='../../data/coco_pyp/MSCOCO'
dataType_train='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType_train)
coco_train, cats, cat_ids = read_data_from_path (dataDir, dataType_train)

dataType_val='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType_val)
coco_val, cats, cat_ids = read_data_from_path (dataDir, dataType_val)

img_ids_train = get_all_image_ids (coco_train)
img_ids_val = get_all_image_ids (coco_val)

img_ids_all = img_ids_train + img_ids_val

# this is valid for both validation or train sets
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco_train, change_names = True)

catnames_list = []
for key, value in cats_id_to_name.items():
    catnames_list.append(value)
    
#%% step 1


def find_image_unique_labels (imID, coco):
    annId_img = coco.getAnnIds( imgIds=imID, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    unique_objects_ids = []
    for item in anns_image:
        k = item['category_id']
        #m = cats_id_to_name[k]
        if k not in unique_objects_ids:
            unique_objects_ids.append(k)
            
    return unique_objects_ids       

def find_image_path (imID, coco, dataType):
    img = coco.loadImgs(imID)[0]      
    name = img ['file_name']
    imPath = os.path.join(dataType,name )
    # imFullPath = os.path.join(dataDir, dataType_train,name )
    # image = cv2.imread(imFullPath)        
    # plt.imshow(image)
    return imPath

def find_dict_image_to_label (coco, dataType, img_ids ):
    dict_img_id_to_path = {}
    dict_img_path_to_id = {}
    dict_image_to_label = {}
    
    for ind in range(len(img_ids)):
        imID = img_ids[ind]
        # print (ind)
        # print(img_ids)
        unique_objects_ids = find_image_unique_labels (imID, coco)  
        dict_image_to_label [imID] = unique_objects_ids
        
        imPath = find_image_path (imID, coco, dataType)
        dict_img_id_to_path [imID] = imPath
        dict_img_path_to_id [imPath] = imID
    return dict_image_to_label, dict_img_id_to_path, dict_img_path_to_id

# For the train split

coco = coco_train
dataType = dataType_train
img_ids = img_ids_train    
dict_image_to_label_train, dict_img_id_to_path_train, dict_img_path_to_id_train = find_dict_image_to_label (coco, dataType,img_ids ) 

# For the validation split

coco = coco_val
dataType = dataType_val
img_ids = img_ids_val   
dict_image_to_label_val, dict_img_id_to_path_val, dict_img_path_to_id_val = find_dict_image_to_label (coco, dataType,img_ids ) 

# merging train and val

dict_image_to_label_all = dict_image_to_label_train #{**dict_image_to_label_train, **dict_image_to_label_val}
  
dict_img_id_to_path_all = {**dict_img_id_to_path_train , **dict_img_id_to_path_val}
dict_img_path_to_id_all = {**dict_img_path_to_id_train , **dict_img_path_to_id_val}

    
#%% steps 2 and 3

def sort_object (input_dict, reverse):
    objects = list(input_dict.keys())
    values = list (input_dict.values())
    if reverse:
        sorted_ind = np.argsort(values)[::-1]
    else:
        sorted_ind = np.argsort(values)
    objects_sorted = [objects[i] for i in sorted_ind ]
    values_sorted = [values[j] for j in sorted_ind]  
    return sorted_ind, objects_sorted,values_sorted 
        

def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0].lower() for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes


def read_captions_from_json():
    audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json'
    with open(audio_dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']
    return data

def find_dict_image_to_captions (data):
    dict_image_id_to_captions = {}
    for item in data:
        image = item ['image']
        cap = item['caption']['text']
        image_id = dict_img_path_to_id_all [image]
        if image_id not in dict_image_id_to_captions:
            dict_image_id_to_captions[image_id] = []
            dict_image_id_to_captions[image_id].append(cap)
        else:
            dict_image_id_to_captions[image_id].append(cap)
       
    return dict_image_id_to_captions 
 
def find_dict_image_to_nouns (dict_image_id_to_captions):
    dict_image_to_nouns = {}
    for key_imID, captionlist in dict_image_id_to_captions.items():
        nounslist = []
        for caption in captionlist:
            cap = caption.lower()
            words = nltk.word_tokenize(cap)
            nouns , noun_indexes = detec_nouns (words) 
            nounslist.append(nouns)
        dict_image_to_nouns [key_imID] = nounslist
    return dict_image_to_nouns

#%% step 2

# We work with "train " data from Karpathy split 

data = read_captions_from_json()
dict_image_id_to_captions = find_dict_image_to_captions (data)

                 
#%% step 3

dict_image_to_nouns = find_dict_image_to_nouns (dict_image_id_to_captions)

#%% automatically getting names using a threshold
#%% 

dict_unique_nouns = {}
for key_imId, value_caplists in dict_image_to_nouns.items():
    for cap in value_caplists:
        for noun in cap:
            if noun not in dict_unique_nouns:
                dict_unique_nouns[noun] = 1
            else:
                dict_unique_nouns[noun] += 1            

sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sort_object (dict_unique_nouns, reverse = True)
m = 1557 # manually selecting only nouns that are repeated at least 10 times and more
sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sorted_ind [0:m], unique_nouns_sorted[0:m], unique_nouns_counts_sorted [0:m]

dict_unique_nouns_freq10 = {}
for counter, n in enumerate(unique_nouns_sorted):
    dict_unique_nouns_freq10[n] = unique_nouns_counts_sorted [counter]


#%% selecting proper names
#%%
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/worktmp2/hxkhkh/current/Dcase/model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)


threshold = 0.5
dict_names_to_nouns = {}
dict_names_to_nouns_counts = {}
dict_names_to_nouns_simple = {}
dict_names_to_nouns_counts_all = {}
error_nouns = [] # only 42 among 4737 most frequent nouns
for noun, count_noun in dict_unique_nouns_freq10.items():
    try:
        sims = [model.similarity (noun, label) for label in catnames_list]
        sim_max = np.max (sims)
        if sim_max >= threshold:
            ind = np.argmax(sims)
            label_max = catnames_list [ind]
            if label_max not in dict_names_to_nouns:
                dict_names_to_nouns[label_max] = {}
                dict_names_to_nouns[label_max][noun] = {}
                dict_names_to_nouns[label_max][noun]['sim'] = sim_max
                dict_names_to_nouns[label_max][noun]['count'] = count_noun
                dict_names_to_nouns_counts[label_max] = {}
                dict_names_to_nouns_counts[label_max][noun] = count_noun
                dict_names_to_nouns_simple [label_max] = []
                dict_names_to_nouns_simple [label_max].append(noun)
                dict_names_to_nouns_counts_all [label_max] = count_noun
            else:
                dict_names_to_nouns[label_max][noun] = {}
                dict_names_to_nouns[label_max][noun]['sim'] = sim_max
                dict_names_to_nouns[label_max][noun]['count'] = count_noun
                dict_names_to_nouns_counts[label_max][noun] = count_noun
                dict_names_to_nouns_simple [label_max].append(noun)
                dict_names_to_nouns_counts_all [label_max] += count_noun
    except:
        print( 'the noun ' + noun + ' is not present ')
        error_nouns.append(noun)

#dict_names_to_nouns_counts_sorted = sorted(dict_names_to_nouns_counts_all.items(), key=lambda x:x[1], reverse=False) 
            
# exceptions:
    # the noun doughnut is not present 
    # the noun doughnuts is not present 
    # the noun grey is not present 
    # the noun selfie is not present 
    
### we should manually add "doughnut" and "doughnuts" ###
for c in catnames_list:
    if c not in dict_names_to_nouns:
        print(c)
        
#%%
# This section asls user for selecting relevant words

# dict_words_sorted = {}
# dict_words_selected = {}
# dict_words_selected_counts = {}
# for label, dict_nouns in dict_names_to_nouns_counts.items():
#     list_sorted = sorted (dict_nouns.items(), key=lambda x:x[1], reverse=True)
#     dict_words_sorted [label] = list_sorted
#     words = []
#     counts = 0
#     print(list_sorted)
#     print('###############')
#     for item in list_sorted:
#         val = input('Is "'+ item[0] + '" a word for label "' + label + '" ? (y/n)  ')
#         if val == 'y':
#             words.append(item[0])
#             counts += item[1]
#     dict_words_selected [label] = words 
#     dict_words_selected_counts [label] = counts       


#%% saving the results

# caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/dict_words_selected.json"
# with open(caption_json, "w") as fp:
#     json.dump(dict_words_selected,fp) 


# caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/dict_words_selected_counts.json"
# with open(caption_json, "w") as fp:
#     json.dump(dict_words_selected_counts,fp)
    
        
# with open(caption_json, 'r') as fp:
#     data_json_test = json.load(fp)

#%% call data from saved files

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/dict_words_selected.json"
with open(caption_json, 'r') as fp:
    dict_words_selected = json.load(fp)

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/dict_words_selected_counts.json"
with open(caption_json, 'r') as fp:
    dict_words_selected_counts = json.load(fp)
   
#%%

# manually filtering words
dict_words_selected_filtered = copy.deepcopy(dict_words_selected)

dict_words_selected_filtered ['baseball'] = ['baseball bat']
dict_words_selected_filtered ['parking'] = ['parking meter']
dict_words_selected_filtered ['wineglass'] = ['wine glass']
dict_words_selected_filtered ['traffic'] = ['traffic light']
dict_words_selected_filtered ['stop'] =  ['stop sign']
dict_words_selected_filtered ['glove'] = ['baseball glove', 'baseball gloves']
dict_words_selected_filtered ['handbag'] = ['purse'] # this should be changed to handbag later
dict_words_selected_filtered ['dryer'] = ['hair dryer']

#%%

frequent_counts_sorted = sorted (dict_words_selected_counts.items(), key=lambda x:x[1], reverse=True)

words_sorted = []
labels_sorted = []
for tuple_word_count in frequent_counts_sorted:   
    labels_sorted.append(tuple_word_count[0])
    words_sorted.append(dict_words_selected_filtered[tuple_word_count[0]])
    
label_word_sorted = []
for counter, l in enumerate(labels_sorted):
    lw_pair = (l, words_sorted[counter])
    label_word_sorted.append(lw_pair)
    
# check "label_word_sorted" with Okko

#%% 
 
     
all_possible_pairs = []
all_possible_pairs_counts = []
for counter, value in enumerate(label_word_sorted):
    pairs_list = []
    label = value [0]
    # list of words
    words = value [1]
    for word in words:
        word = ' ' + word + ' ' 
        word = word.upper()
        label_id = cats_name_to_id [label]
        
        for image_id, list_of_image_labels in dict_image_to_label_all.items():
            if label_id in list_of_image_labels:
                search_captions = dict_image_id_to_captions [image_id]
                for cap in search_captions:
                    if word in cap:
                        pair = (image_id, cap)
                        pairs_list.append(pair)
                        
    all_possible_pairs.append(pairs_list) 
    all_possible_pairs_counts.append(len(pairs_list))              

#%% sorting based on all possible pairs # The main sorting is happening here #

all_possible_pairs_counts_sorted = np.sort(all_possible_pairs_counts)[::-1]
ind_sorted = np.argsort(all_possible_pairs_counts)[::-1]
all_possible_pairs_sorted = [all_possible_pairs[i] for i in ind_sorted]
label_word_final = [label_word_sorted[i] for i in ind_sorted]

# you should work with "label_word_final"


#%% 
from sklearn.utils import shuffle

     
pool_all_organized = []
pool_all = {}

for ind, pool_lw in enumerate(all_possible_pairs_sorted):
    pool_shuffled = shuffle(pool_lw, random_state=0)   
    pool_all_organized.append(pool_shuffled)

    for item in pool_shuffled:
        if item not in pool_all:
            pool_all[item] = []
            pool_all[item].append(ind)
        else:
            pool_all[item].append(ind)
            
#%% RWS

# 2 months to 4 months of simulation  (or going to 6 months and see when the learning starts to happen)
#######        simulatin the language experinece       ########
kh
###############################   input values ################################
##########################   select proper subset #############################  
subset_name = 'subset4'
    
simulation_days = 240 # days
minutes_per_day = 56.1
beta = 1 # co-occurrence factor

###############################
subset_name = 'subset3'
    
simulation_days = 240 # days
minutes_per_day = 56.1
beta = 0.75 # co-occurrence factor

###############################
subset_name = 'subset2'
    
simulation_days = 120 # days
minutes_per_day = 56.1
beta = 1 # co-occurrence factor

###############################
subset_name = 'subset1'
    
simulation_days = 120 # days
minutes_per_day = 56.1
beta = 0.75 # co-occurrence factor


###############################
subset_name = 'subset0'
    
simulation_days = 120 # days
minutes_per_day = 56.1
beta = 0.50 # co-occurrence factor


#%%
total_time = (1/60) * simulation_days * minutes_per_day # hours
total_time_co_occurrence = beta * total_time 
print(total_time_co_occurrence)

###############################################################################


#######    real word statistic for namings frequencies ( # / hour)    ########

from scipy.io import loadmat
file_name = save_path + 'rws_counts_sorted.mat'
rws_sorted = loadmat(file_name, variable_names = 'data')
rws_data = rws_sorted['data'][0]   
rws_data_short = rws_data[0:80]
rws_data_unique = []
for item in rws_data:
    if item not in rws_data_unique:
        rws_data_unique.append(item)

phi = rws_data_short

##################

total_co_occurrence = total_time_co_occurrence * phi 
total_co_occurrence_rounded = [int(i) for i in np.ceil(total_co_occurrence)]

# this is comparable with "labels_sorted" and "words_sorted"
# We find N <= 104 images corresponding to those captions, reusing the same image for as many captions as possible. 
# In the end, we have 104 unique image-caption pairs (which can reuse the same image) with “person” bounded box, and “man” spoken in the caption. 
dict_rws = {}
for ind_label, tco in enumerate(total_co_occurrence_rounded):
    dict_rws [ind_label] = tco


#%%
# iterative algorithm 
# for selecting pairs for each category considering overlapping categories
dict_selected_pairs = {}
dict_selected_stat = {}

for key_pair, list_ind in pool_all.items():
    
    pass_status = True
    # first check if we can accept this pair
    for ind in list_ind: 
        if ind in dict_selected_stat and dict_selected_stat[ind] >= dict_rws [ind] :
           pass_status = False 
           
    if pass_status:
        for ind in list_ind:   
            if ind not in dict_selected_stat:
                dict_selected_pairs[ind] = []
                dict_selected_pairs[ind].append(key_pair)
                dict_selected_stat[ind] = 1
            else:
                dict_selected_pairs[ind].append(key_pair)
                dict_selected_stat[ind] += 1
        
    
#%%    
# check if numer of needed pairs in each category are provided
# there are very few cases of mismatch and that's because of repeating pairs
# this hapens because some namings are repeated twice in one sentence 
# this will be solved at the next step at unifying the pairs as a unique list
mismatch_cases = []
s = 0
for key_ind in dict_selected_stat:
    s += dict_selected_stat[key_ind]
    if dict_selected_stat[key_ind] !=  dict_rws[key_ind]:
        mismatch_cases.append(key_ind)


#%% 
# unifying the pairs as a unique list
# the length of the final list is smaller than s because of the overlapping cases

pool_selected = []
for key, value in dict_selected_pairs.items():
    for pair in value:
        if pair not in pool_selected:
            pool_selected.append(pair)


#%% 
# converting pool and data to dictionaries to make the search faster

dict_pool_selected = {}
for p in pool_selected:
    imID = p[0]
    if imID not in dict_pool_selected:
        dict_pool_selected [imID] = []
        dict_pool_selected [imID].append(p[1])
    else:
        dict_pool_selected [imID].append(p[1])

dict_data = {}
for d in data:
    caption_d = d['caption']['text']
    im = d['image']
    imID= dict_img_path_to_id_all[im]
    if imID not in dict_data:
        dict_data[imID] = []
        dict_data[imID].append(caption_d)
    else:
        dict_data[imID].append(caption_d)

#%%    
data_subset = []    
for d in data:
    caption_d = d['caption']['text']
    im = d['image']
    imID = dict_img_path_to_id_all[im]
    if imID in dict_pool_selected:
        candidate_caps = dict_pool_selected[imID]
        if caption_d in candidate_caps:
            data_subset.append(d)
    

#%%
data_json_subset = {}
data_json_subset ['data'] = data_subset

file_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, "w") as fp:
    json.dump(data_json_subset,fp) 


#%%
# testing 
import json
    
subset_name = 'subset4'

file_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/subsets/SpokenCOCO_train_" + subset_name +  ".json"
with open(file_json, 'r') as fp:
    data_json_test = json.load(fp)
    
data_subset_test = data_json_test['data']
    
print(len(data_subset_test))

#%%

















