import os
import json
from utilsMSCOCO import read_data_from_path, get_all_image_ids, get_all_cats
import cv2
from matplotlib import pyplot as plt
import nltk
import numpy as np

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp2/hxkhkh/current/Dcase/model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
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
kh
#%%

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

def plot_dist_cats (input_dict, save_name, title):
    sorted_ind, objects_sorted,values_sorted = sort_object (input_dict)
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
        plt.savefig(save_name, format='pdf')
        

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

def find_dict_image_to_names (dict_image_to_nouns, dict_nouns_to_names):
    dict_image_to_names = {}
    for key_imID, values in dict_image_to_nouns.items():
        values_new = []
        for nounlist in values:
            namelist = []
            for n in nounlist:
                if n in dict_nouns_to_names:
                    namelist.append(dict_nouns_to_names[n])
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
#%% step 3

dict_image_to_nouns = find_dict_image_to_nouns (dict_image_to_captions)

#%% automatically getting names using a threshold
#%% 
mydict = {}


unique_nouns = {}
for key_imId, value_caplists in dict_image_to_nouns.items():
    for cap in value_caplists:
        for n in cap:
            if n not in unique_nouns:
                unique_nouns[n] = 1
            else:
                unique_nouns[n] += 1
            # if n=='.jpg':
            #     print(cap)
            #     print(dict_image_to_captions[key_imId])




sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sort_object (unique_nouns, reverse = True)
n = 4743 # manually selecting only nouns that are repeated at least 10 times and more
sorted_ind, unique_nouns_sorted, unique_nouns_counts_sorted  = sorted_ind [0:4743], unique_nouns_sorted[0:4743], unique_nouns_counts_sorted [0:4743]

unique_nouns_frequent = {}
for counter, n in enumerate(unique_nouns_sorted):
    unique_nouns_frequent[n] = unique_nouns_counts_sorted [counter]

mydict = {}
for noun, count_noun in unique_nouns_frequent.items():
    try:
        sims = [model.similarity (noun, label) for label in catnames_list]
        sim_max = np.max (sims)
        if sim_max >= 0.5:
            ind = np.argmax(sims)
            label_max = catnames_list [ind]
            if label_max not in mydict:
                mydict[label_max] = {}
                mydict[label_max][noun] = {}
                mydict[label_max][noun]['sim'] = sim_max
                mydict[label_max][noun]['count'] = count_noun
            else:
                mydict[label_max][noun] = {}
                mydict[label_max][noun]['sim'] = sim_max
                mydict[label_max][noun]['count'] = count_noun
    except:
        print( 'the noun ' + noun + ' is not present ')
            
    
    
# p = '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/names/names_all_nouns_0.5.mat'
# import scipy
# mfile = scipy.io.loadmat(p)
# a = sorted(mfile.keys())[3:]
# mydict = {}
# for key in a:
#     mydict[key] = mfile[key]


#%% manually selecting proper names


for key_label, valuelist in mydict.items():
    sim = [model.similarity(str(name.strip()), key_label) for name in valuelist]

#%%    
dict_nouns_to_names = {}
for key, nounlist in mydict.items():
    for noun in nounlist:
        print (noun)
        n = str(noun)
        n_key = n.strip()
        dict_nouns_to_names[n_key] = key

#%%

dict_image_to_names = find_dict_image_to_names (dict_image_to_nouns, dict_nouns_to_names)

#%%
counts_namings_labelsID = {}
for key_imID, value_namelists in dict_image_to_names.items():
    labelIDs = dict_image_to_label_all[key_imID]
    namings_5caps = dict_image_to_names [key_imID]
    count = 0
    
    for labelID in labelIDs:
        
        label = cats_id_to_name [labelID]
        if labelID not in counts_namings_labelsID:
            counts_namings_labelsID [labelID] = []
            
        count = 0    
        for namings_eachcap in namings_5caps:
            if label in namings_eachcap:
                count += 1
        
        counts_namings_labelsID [labelID].append(count)


frequency_namings_labels = {}        
for labelID , value in counts_namings_labelsID.items():
    newKey = cats_id_to_name [labelID]
    frequency_namings_labels [newKey] = round (np.mean(value),2)
    
        
#%%



        
save_name = '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/frequency_labels.pdf'
title = 'average number of naming events for each label'
plot_dist_cats (frequency_namings_labels, save_name, title)