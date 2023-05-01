import os
import json
from utilsMSCOCO import read_data_from_path, get_all_image_ids, get_all_cats
import cv2
from matplotlib import pyplot as plt
import nltk
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
#%%


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

#%%
def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0] for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes

# def get_unique_nouns (data_json):
#     wavfile_nouns = []
#     unique_nouns = {}
#     for k in range(len(data_json['annotations'])):
#         data_annFile_example = data_json['annotations'][k]
#         caption_example = data_annFile_example['caption']
        
#         words = nltk.word_tokenize(caption_example)
         
#         nouns , noun_indexes = detec_nouns (words)
#         for n in nouns:
#             if n not in unique_nouns:
#                 unique_nouns[n] = 1
#             else:
#                 unique_nouns[n] += 1
#         wavfile_nouns.append(nouns)
#     return unique_nouns, wavfile_nouns

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
dict_image_to_nouns = find_dict_image_to_nouns (dict_image_to_captions)

#%%
p = '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/names/names_all_nouns_0.5.mat'
import scipy
mfile = scipy.io.loadmat(p)
a = sorted(mfile.keys())[3:]
mydict = {}
for key in a:
    mydict[key] = mfile[key]
    
dict_nouns_to_names = {}
for key, nounlist in mydict.items():
    for noun in nounlist:
        print (noun)
        n = str(noun)
        n_key = n.strip()
        dict_nouns_to_names[n_key] = key

#%%

dict_image_to_names = find_dict_image_to_names (dict_image_to_nouns, dict_nouns_to_names)
