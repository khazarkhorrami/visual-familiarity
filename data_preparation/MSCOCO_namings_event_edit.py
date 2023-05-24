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
    img_id_to_path = {}
    img_path_to_id = {}
    dict_image_to_label = {}
    
    for ind in range(len(img_ids)):
        imID = img_ids[ind]
        # print (ind)
        # print(img_ids)
        unique_objects_ids = find_image_unique_labels (imID, coco)  
        dict_image_to_label [imID] = unique_objects_ids
        
        imPath = find_image_path (imID, coco, dataType)
        img_id_to_path [imID] = imPath
        img_path_to_id [imPath] = imID
    return dict_image_to_label, img_id_to_path, img_path_to_id

# For the train split

coco = coco_train
dataType = dataType_train
img_ids = img_ids_train    
dict_image_to_label_train, img_id_to_path_train, img_path_to_id_train = find_dict_image_to_label (coco, dataType,img_ids ) 

# For the validation split

coco = coco_val
dataType = dataType_val
img_ids = img_ids_val   
dict_image_to_label_val, img_id_to_path_val, img_path_to_id_val = find_dict_image_to_label (coco, dataType,img_ids ) 

# merging train and val

dict_image_to_label_all = dict_image_to_label_train #{**dict_image_to_label_train, **dict_image_to_label_val}
  
img_id_to_path_all = {**img_id_to_path_train , **img_id_to_path_val}
img_path_to_id_all = {**img_path_to_id_train , **img_path_to_id_val}


# saving the results
# since dict_image_to_label_all cannot be saved as mat file, I provided another dictionary with keys equal to image paths.

dict_imagepath_to_label_all = {}
for imID, value in dict_image_to_label_all.items():
    impath = img_id_to_path_all [imID]
    dict_imagepath_to_label_all [impath] = value
    
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
        image_id = img_path_to_id_all [image]
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
kh
dict_frequent_words = {}
dict_frequent_words_counts = {}

for label, dict_nouns in dict_names_to_nouns_counts.items():
    list_sorted = sorted (dict_nouns.items(), key=lambda x:x[1], reverse=True)
    frequent_noun_tuple = list_sorted[0]
    dict_frequent_words[label] = frequent_noun_tuple [0]
    dict_frequent_words_counts[label] = frequent_noun_tuple [1]
    
frequent_counts_sorted = sorted (dict_frequent_words_counts.items(), key=lambda x:x[1], reverse=True)

words_sorted = []
labels_sorted = []
for tuple_word_count in frequent_counts_sorted:   
    labels_sorted.append(tuple_word_count[0])
    words_sorted.append(dict_frequent_words[tuple_word_count[0]])
    
label_word_sorted = []
for counter, l in enumerate(labels_sorted):
    lw_pair = (l, words_sorted[counter])
    label_word_sorted.append(lw_pair)
    
#%%

# manually filtering words
label_word_filtered = copy.deepcopy(label_word_sorted)

label_word_filtered [6] = ('baseball', 'baseball bat')
label_word_filtered [7] = ('toilet','toilet')
label_word_filtered [9] = ('refrigerator','refrigerator' )
label_word_filtered [18] = ('laptop', 'laptop')
label_word_filtered [26] = ('parking', 'parking meter')
label_word_filtered [34] = ( 'wineglass', 'wine glass')
label_word_filtered [36] = ('traffic', 'traffic light')
label_word_filtered [44] = ('broccoli','broccoli')
label_word_filtered [50] = ('suitcase', 'suitcase')
label_word_filtered [51] = ('bottle', 'bottle')
label_word_filtered [54] = ('apple', 'apple')
label_word_filtered [57] =  ('stop', 'stop sign')
label_word_filtered [58] = ('oven', 'oven')
label_word_filtered [61] = ('spoon', 'spoon')
label_word_filtered [65] = ('backpack', 'backpack')
label_word_filtered [75] = ('glove', 'baseball glove')
label_word_filtered [76] = ('handbag', 'purse') # this should be changed to handbag later
label_word_filtered [79] = ('dryer', 'hair dryer') 

#%%   
all_possible_pairs = []
all_possible_pairs_counts = []
for counter, value in enumerate(label_word_filtered):
    pairs_list = []
    label = value [0]
    word = ' ' + value [1] + ' ' 
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

#%% sorting based on all possible pairs

all_pairs_counts_sorted = np.sort(all_possible_pairs_counts)[::-1]
ind_sorted = np.argsort(all_possible_pairs_counts)[::-1]
all_pairs_sorted = [all_possible_pairs[i] for i in ind_sorted]
label_word_final = [label_word_filtered[i] for i in ind_sorted]

#%% 

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

#%% 

# 2 months to 4 months of simulation  (or going to 6 months and see when the learning starts to happen)
#######        simulatin the language experinece       ########

simulation_days = 240 # days
minutes_per_day = 56.1
beta = 1 # co-occurrence factor

total_time = (1/60) * simulation_days * minutes_per_day # hours
total_time_co_occurrence = beta * total_time 

total_co_occurrence = total_time_co_occurrence * phi 
total_co_occurrence_rounded = [int(i) for i in np.ceil(total_co_occurrence)]

# this is comparable with "labels_sorted" and "words_sorted"
# We find N <= 104 images corresponding to those captions, reusing the same image for as many captions as possible. 
# In the end, we have 104 unique image-caption pairs (which can reuse the same image) with “person” bounded box, and “man” spoken in the caption. 

    
#%% 
from sklearn.utils import shuffle

     
pool_all_organized = []
pool_all_organized_selection = []
pool_all = {}

for ind, pool_lw in enumerate(all_pairs_sorted):
    pool_shufled = shuffle(pool_lw, random_state=0)
    selection = pool_shufled [0:total_co_occurrence_rounded [ind]]
    pool_all_organized.append(pool_shufled)
    pool_all_organized_selection.append(selection)
    for item in pool_shufled:
        if item not in pool_all:
            pool_all[item] = []
            pool_all[item].append(ind)
        else:
            pool_all[item].append(ind)

pool_all_unique = {}      
for key, value in pool_all.items():
    if len(value) <= 1:
        pool_all_unique[key] = value

pool_all_descending = pool_all_organized_selection [::-1]

#%%
dict_frequencies = {}
for ind, f in enumerate(total_co_occurrence_rounded):
    dict_frequencies [ind] = f
    
selected_pairs = {}
selected_pairs_all = [] # 11586
for candidate_pair , value_ind_list in pool_all_unique.items(): 
    ind = value_ind_list[0]
    freq = dict_frequencies[ind] 
    if ind not in selected_pairs:
        selected_pairs[ind] = []
        selected_pairs[ind].append(candidate_pair)
        selected_pairs_all.append(candidate_pair) 
    elif len (selected_pairs[ind]) < freq:
        selected_pairs[ind].append(candidate_pair)
        selected_pairs_all.append(candidate_pair) 
       
#%%

test = selected_pairs[79]
word = ' '+ 'phone' + ' '
for i_tuple in test:
    cap = i_tuple[1]
    if word in cap:
        print(cap)
        
for key, list_i in selected_pairs.items():
    if len(list_i) != dict_frequencies[key]:
        print("there is a mismatch in " )
        print(key)

#%%
        
#unifying all selected pairs to include in train data

data_subset = []    
for data_pair in data:
    imID = data_pair['image_id']
    caption = data_pair['caption']
    if (imID, caption) in selected_pairs_all:
        data_subset.append(data_pair)

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/annotations/captions_train2014.json"
with open(caption_json, 'r') as fp:
    data_json = json.load(fp)
data_annotations = data_json['annotations']
data_json['annotations'] = data_subset
# data_images = data_json['images']
# data_json['images'] = []

#%%
caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/captions_train2014_subset4.json"
with open(caption_json, "w") as fp:
    json.dump(data_json,fp) 
    
caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/captions_train2014_subset4.json"
with open(caption_json, 'r') as fp:
    data_json_test = json.load(fp)
#%%




















