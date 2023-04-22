import numpy
import cv2
import os
from matplotlib import pyplot as plt
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids, read_image_anns


dataDir='../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)




image_id = img_ids [3800]
img = coco.loadImgs(image_id)[0]
h = img ['height']
w = img ['width']
name = img ['file_name']
imPath = os.path.join(dataDir, dataType,name )
image = cv2.imread(imPath)
annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
anns_image = coco.loadAnns(annId_img)
mask_annitem = numpy.zeros([h,w])
for item in anns_image : # constructing true mask by ading all mask items
    mask_temp = coco.annToMask(item)
    mask_annitem = mask_annitem + mask_temp

plt.imshow(mask_temp)
plt.imshow(image)


img, anns_image, items = read_image_anns (image_id, coco)

import json
from nltk.tokenize import word_tokenize
import nltk

data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"
audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']

example_cap = data [0]['caption']['text']

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO/annotations/captions_val2014.json"
with open(caption_json, 'r') as fp:
    data_json = json.load(fp)


    
    
def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    nouns = [n[0] for n in (tok) if n[1] =='NN' or n[1] =='NNS' ]
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return nouns , noun_indexes

wavfile_nouns = []
unique_nouns = []
for k in range(len(data_json['annotations'])):
    data_annFile_example = data_json['annotations'][k]
    caption_example = data_annFile_example['caption']
    
    words = word_tokenize(caption_example)
     
    nouns , noun_indexes = detec_nouns (words)
    for n in nouns:
        if n not in unique_nouns:
            unique_nouns.append(n)
    
    # print(caption_example)
    # print(words)
    # print(nouns)
    # print ('..................................................................')
    
    wavfile_nouns.append(nouns)
    
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/worktmp2/hxkhkh/current/Dcase/model/word2vec/GoogleNews-vectors-negative300.bin', binary=True)


ind_accepted = []    
captions_rejected = []
for counter_utterance, candidate_utterance  in enumerate(wavfile_nouns[0:200]):        
    print(counter_utterance)
    try:
        model.most_similar(candidate_utterance)
        ind_accepted.append(counter_utterance)
    except:
        captions_rejected.append(candidate_utterance)
        print("An exception occurred in word:  " + str(candidate_utterance))
all_labels = []
cats_id_to_name[10] = 'pilot'
cats_id_to_name[11] = 'hydrant'
cats_id_to_name[13] = 'stop'
cats_id_to_name[14] = 'parking'
cats_id_to_name[37] = 'ball'
cats_id_to_name[39] = 'baseball'
cats_id_to_name[40] = 'glove'
cats_id_to_name[43] = 'tennis'
cats_id_to_name[46] = 'glass'
cats_id_to_name[58] = 'sausage'
cats_id_to_name[64] = 'plant'
cats_id_to_name[67] = 'table'
cats_id_to_name[77] = 'phone'
cats_id_to_name[88] = 'teddy'
cats_id_to_name[89] = 'drier'
for counter, label in cats_id_to_name.items():
    print(label) 
    sim = model.similarity('man',label) 
    all_labels.append(label)


all_sims = []     
for un in unique_nouns:
    print(un)
    try:
        sim = [model.similarity(un, label) for label in all_labels]
        all_sims.extend(sim)
    except:
        pass
    if max(sim) >= 0.4:
        l = all_labels [numpy.argmax(sim)]
        print(un + ' is very similar to ' + l)

from matplotlib import pyplot as plt
plt.hist(all_sims , bins = 100) 