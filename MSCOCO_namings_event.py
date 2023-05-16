import os
import json
import scipy
from utilsMSCOCO import read_data_from_path, get_all_image_ids, get_all_cats
import cv2
from matplotlib import pyplot as plt
import nltk
import numpy as np

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp2/hxkhkh/current/Dcase/model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

save_path =   '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/new/'  

#%%
######################### reading train and val data

dataDir='../data/coco_pyp/MSCOCO'
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

dict_image_to_label_all = {**dict_image_to_label_train, **dict_image_to_label_val}
  
img_id_to_path_all = {**img_id_to_path_train , **img_id_to_path_val}
img_path_to_id_all = {**img_path_to_id_train , **img_path_to_id_val}


# saving the results
# since dict_image_to_label_all cannot be saved as mat file, I provided another dictionary with keys equal to image paths.

dict_imagepath_to_label_all = {}
for imID, value in dict_image_to_label_all.items():
    impath = img_id_to_path_all [imID]
    dict_imagepath_to_label_all [impath] = value
    
# save_name = save_path + 'dict_imagepath_to_label_all.mat'
# scipy.io.savemat(save_name, dict_imagepath_to_label_all)

# save_name = save_path + 'img_path_to_id_all.mat'
# scipy.io.savemat(save_name, img_path_to_id_all)


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

def plot_dist_cats (input_dict, save_name, title, f):
    sorted_ind, objects_sorted,values_sorted = sort_object (input_dict, reverse = True)
    fig, ax = plt.subplots(figsize = (16,16))
    ax.barh(objects_sorted, values_sorted)   
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)        
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)     
    # Show top values
    ax.invert_yaxis()   
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')
     
    # Add Plot Title
    ax.set_title(title,
                 loc ='center', )
    if save_name:
        plt.savefig(save_name, format=f)
        

def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0].lower() for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes


def read_captions_from_json(split):
    caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/annotations/captions_" + split + "2014.json"
    with open(caption_json, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['annotations']
    # caption_example = data [0]['caption'] # 0:414113
    # number_of_captions = len (data)
    return data

def find_dict_image_to_captions (data):
    dict_image_to_captions = {}
    
    for counter, d in enumerate(data):
        imID = d ['image_id']
        cap = d ['caption']
        if imID not in dict_image_to_captions:
            dict_image_to_captions [imID] = []
            dict_image_to_captions [imID].append(cap)
        else:
            dict_image_to_captions [imID].append(cap)
            
    return dict_image_to_captions
 
def find_dict_image_to_nouns (dict_image_to_captions):
    dict_image_to_nouns = {}
    for key_imID, captionlist in dict_image_to_captions.items():
        nounslist = []
        for cap in captionlist:
            words = nltk.word_tokenize(cap)
            nouns , noun_indexes = detec_nouns (words) 
            nounslist.append(nouns)
        dict_image_to_nouns [key_imID] = nounslist
    return dict_image_to_nouns

def find_dict_image_to_names (dict_image_to_nouns, dict_noun_to_name):
    dict_image_to_names = {}
    for key_imID, caps_nouns in dict_image_to_nouns.items():
        values_new = []
        for cap_nouns in caps_nouns:
            namelist = []
            for n in cap_nouns:
                if n in dict_noun_to_name:
                    namelist.append(dict_noun_to_name[n])
            values_new.append(namelist)
        dict_image_to_names[key_imID] = values_new
    return dict_image_to_names

#%% step 2
split = 'train'
data_train = read_captions_from_json(split)
# dict_image_to_captions_train = find_dict_image_to_captions (data_train)
# dict_image_to_nouns_train = find_dict_image_to_nouns (dict_image_to_captions_train)

split = 'val'
data_val = read_captions_from_json(split)
# dict_image_to_captions_val = find_dict_image_to_captions (data_val)
# dict_image_to_nouns_val = find_dict_image_to_nouns (dict_image_to_captions_val)

# for train + val
data = data_train + data_val
dict_image_to_captions = find_dict_image_to_captions (data)

# saving the results
save_name = save_path + 'dict_imagepath_to_captions.mat'

dict_imagepath_to_captions = {}
for imID, value in dict_image_to_captions.items():
    impath = img_id_to_path_all [imID]
    dict_imagepath_to_captions [impath] = value
    
#scipy.io.savemat(save_name, dict_imagepath_to_captions)
                 
#%% step 3

dict_image_to_nouns = find_dict_image_to_nouns (dict_image_to_captions)

# saving the results
save_name = save_path + 'dict_imagepath_to_nouns.mat'

dict_imagepath_to_nouns = {}
for imID, value in dict_image_to_nouns.items():
    impath = img_id_to_path_all [imID]
    dict_imagepath_to_nouns [impath] = value
    
#scipy.io.savemat(save_name, dict_imagepath_to_nouns)

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
            # if n=='.jpg':
            #     print(cap)
            #     print(dict_image_to_captions[key_imId])




sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sort_object (dict_unique_nouns, reverse = True)
m = 4743 # manually selecting only nouns that are repeated at least 10 times and more
sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sorted_ind [0:m], unique_nouns_sorted[0:m], unique_nouns_counts_sorted [0:m]

dict_unique_nouns_freq10 = {}
for counter, n in enumerate(unique_nouns_sorted):
    dict_unique_nouns_freq10[n] = unique_nouns_counts_sorted [counter]
   
# plotting distribution of 100 frequent nouns
# m = 100
# sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sorted_ind [0:m], unique_nouns_sorted[0:m], unique_nouns_counts_sorted [0:m]
# dict_unique_nouns_100 = {}
# for counter, n in enumerate(unique_nouns_sorted):
#     dict_unique_nouns_100[n] = unique_nouns_counts_sorted [counter]
    
# save_name = save_path + 'dist_nouns_100.pdf'
# title = 'distribution of 100 most frequent nouns '
# plot_dist_cats (dict_unique_nouns_100, save_name, title)
kh
#%% manually selecting proper names
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

dict_names_to_nouns_counts_sorted = sorted(dict_names_to_nouns_counts_all.items(), key=lambda x:x[1], reverse=False) 
             
dict_names_counts_all = {}
dict_names_sims = {}
dict_names_variety = {}
for key_label, value in dict_names_to_nouns.items():
    for key_name, info_name in value.items():
        if key_label not in dict_names_counts_all:
            dict_names_counts_all [key_label] = info_name ['count']
            dict_names_sims [key_label] = []
            dict_names_sims [key_label].append(info_name ['sim'])
            dict_names_variety[key_label] = 1
        else:
            dict_names_counts_all [key_label] += info_name ['count']
            dict_names_sims [key_label].append(info_name ['sim'])
            dict_names_variety[key_label] += 1
            
#Plotting the distribution of names
save_name = save_path + 'dist_names_05.png'
title = 'distribution of all name counts '
f = 'png'
plot_dist_cats (dict_names_counts_all, save_name, title, f)

save_name = save_path + 'dist_names_variabilty_05.png'
title = 'distribution of all name variabilities '
f = 'png'
plot_dist_cats (dict_names_variety, save_name, title, f)

#%%

dict_names_frequent = {}
dict_names_frequent_words = {}
dict_names_frequent_counts = {}

for label, dict_nouns in dict_names_to_nouns_counts.items():
    list_sorted = sorted (dict_nouns.items(), key=lambda x:x[1], reverse=True)
    frequent_noun_tuple = list_sorted[0]
    dict_names_frequent[label] = frequent_noun_tuple
    dict_names_frequent_words[label] = frequent_noun_tuple [0]
    dict_names_frequent_counts[label] = frequent_noun_tuple [1]
    
frequent_counts_sorted = sorted (dict_names_frequent_counts.items(), key=lambda x:x[1], reverse=True)
#%% both methods are same, the second one is more general

# dict_noun_to_name = {}

# for key_name, noun_list in dict_names_to_nouns.items():
#     for n in noun_list:
#         if n not in dict_noun_to_name:
#             dict_noun_to_name [n] = key_name
            
dict_noun_to_name = {}            
for noun, count_noun in dict_unique_nouns_freq10.items():            
    try:
        sims = [model.similarity (noun, label) for label in catnames_list]
        sim_max = np.max (sims)
        if sim_max >= threshold:
            ind = np.argmax(sims)
            label_max = catnames_list [ind]
            dict_noun_to_name [noun] = label_max
    except:
        print( 'the noun ' + noun + ' is not present ')
        error_nouns.append(noun)
#%%

dict_image_to_names = find_dict_image_to_names (dict_image_to_nouns, dict_noun_to_name)

#%%
counts_namings_labelsID = {}
for key_imID, value_namelists in dict_image_to_names.items():
    labelIDs = dict_image_to_label_all[key_imID]
    namings_5caps = dict_image_to_names [key_imID]
    
    
    for labelID in labelIDs:
        
        label = cats_id_to_name [labelID]
        if labelID not in counts_namings_labelsID:
            counts_namings_labelsID [labelID] = []
            
        count = 0    
        for namings_eachcap in namings_5caps:
            if label in namings_eachcap:
                count += 1
        #print (count)
        counts_namings_labelsID [labelID].append(count)


frequency_namings_labels = {}        
for labelID , value in counts_namings_labelsID.items():
    newKey = cats_id_to_name [labelID]
    frequency_namings_labels [newKey] = round (np.mean(value),2)
    
kh
#%%
    
save_name = save_path + 'frequency_labels_09.png'
title = 'average number of naming events for each label'
f = 'png'
plot_dist_cats (frequency_namings_labels, save_name, title, f)

#%%
number_of_images_per_label = {}

for key_naming, value in frequency_namings_labels.items():
    number_of_images_per_label [key_naming] =  dict_names_counts [key_naming] / value
        
#%%
    
save_name = save_path + 'image_numbers_needed_09.png'
title = 'average number of images for each naming event'
f = 'png'
plot_dist_cats (number_of_images_per_label, save_name, title, f)

#%%
from nltk.corpus import wordnet as wn

test =  wn.synsets('apple')
print(test)

print(wn.synset('apple.n.02').definition())

















