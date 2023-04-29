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

cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco_train)
#%%
img_id_to_path = {}
img_path_to_id = {}
ind = 10
imID = img_ids_train[ind] 
img = coco_train.loadImgs(imID)[0]

annId_img = coco_train.getAnnIds( imgIds=imID, iscrowd=False) 
anns_image = coco_train.loadAnns(annId_img)

name = img ['file_name']
imPath = os.path.join(dataType_train,name )
imFullPath = os.path.join(dataDir, dataType_train,name )
image = cv2.imread(imFullPath)
annId_img = coco_train.getAnnIds( imgIds=imID, iscrowd=False) 
anns_image = coco_train.loadAnns(annId_img)
unique_objects_ids = []
for item in anns_image:
    k = item['category_id']
    m = cats_id_to_name[k]
    print(m)
    if k not in unique_objects_ids:
        unique_objects_ids.append(k)
        
plt.imshow(image)
print (len(annId_img))
print('#######')
print([cats_id_to_name [m] for m in unique_objects_ids])

dict_image_to_label = {}
dict_image_to_label [imID] = unique_objects_ids

img_id_to_path [imID] = imPath
img_path_to_id [imPath] = imID
#%%
def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0] for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes

def get_unique_nouns (data_json):
    wavfile_nouns = []
    unique_nouns = {}
    for k in range(len(data_json['annotations'])):
        data_annFile_example = data_json['annotations'][k]
        caption_example = data_annFile_example['caption']
        
        words = nltk.word_tokenize(caption_example)
         
        nouns , noun_indexes = detec_nouns (words)
        for n in nouns:
            if n not in unique_nouns:
                unique_nouns[n] = 1
            else:
                unique_nouns[n] += 1
        wavfile_nouns.append(nouns)
    return unique_nouns, wavfile_nouns

def read_captions_from_json(split):
    caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/annotations/captions_" + split + "2014.json"
    with open(caption_json, 'r') as fp:
        data_json = json.load(fp)
    unique_nouns, wavfile_nouns = get_unique_nouns (data_json)
    #unique_nouns_sorted = sorted(unique_nouns.items(), key=lambda x:x[1], reverse=True)   
    return unique_nouns, wavfile_nouns

split = 'train'
unique_nouns_train, wavfile_nouns_train = read_captions_from_json(split)
dict_image_to_captions = {}
dict_image_to_nouns = {}
 
#%%
p = '/worktmp2/hxkhkh/current/FaST/plots/vf/distributions/names/names_all_nouns_0.5.mat'
import scipy
mfile = scipy.io.loadmat(p)
a = sorted(mfile.keys())[3:]
mydict = {}
for key in a:
    mydict[key] = mfile[key]