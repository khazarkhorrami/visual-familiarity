import numpy as np
import cv2
import os
import json 
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids, change_labels
root = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/"
jsub0 = os.path.join(root, "SpokenCOCO_train_subset0A.json")
jsub1 = os.path.join(root, "SpokenCOCO_train_subset1.json")
jsub2 = os.path.join(root, "SpokenCOCO_train_subset2.json")
jsub3 = os.path.join(root, "SpokenCOCO_train_subset3.json")
#%%

def read_data_subsets(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    return data

def read_images_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    images = []
    for d in data:
        im = d['image']      
        images.append(im)
    return images

# stack all images
images_subs = []
images_sub3 = read_images_json(jsub3)
images_subs.extend(images_sub3)
images_sub0 = read_images_json(jsub0)
for i in images_sub0:
    if i not in images_subs:
        images_subs.append(i)
images_sub1 = read_images_json(jsub1)
for i in images_sub1:
    if i not in images_subs:
        images_subs.appendd(i)
images_sub2 = read_images_json(jsub2)
for i in images_sub2:
    if i not in images_subs:
        images_subs.append(i)

# stack all data
data0 = read_data_subsets(jsub0)
data1 = read_data_subsets(jsub1)
data2 = read_data_subsets(jsub2)
data3 = read_data_subsets(jsub3)

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
saveDir = "../../../semtest/images"
dataDir='../../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)

img_id_to_img_filenames = {}
img_filenames_to_img_id = {}
img_filenames_to_all = {}
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    fn = img['file_name']
    imID = img['id']
    img_id_to_img_filenames[imID] = fn
    img_filenames_to_img_id [fn] = imID
    img_filenames_to_all [fn] = img
    

all_labels, cats_id_to_short_name = change_labels (cats_id_to_name)  
dict_image_to_label_train, dict_img_id_to_path_train, dict_img_path_to_id_train = find_dict_image_to_label (coco, dataType,img_ids )
dict_image_to_label_all = dict_image_to_label_train
#%% copied this for obtaining "label_word_sorted" but it can be accesses also through dict_words_selected

caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected.json"
with open(caption_json, 'r') as fp:
    dict_words_selected = json.load(fp)

caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected_counts.json"
with open(caption_json, 'r') as fp:
    dict_words_selected_counts = json.load(fp)
   
import copy
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
caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_sorted_objects.json"
with open(caption_json, 'r') as fp:
    labels_sorted = json.load(fp) ['sorted_object_labels']
    
label_word_sorted = []
for counter, l in enumerate(labels_sorted):
    lw_pair = (l, words_sorted[counter])
    label_word_sorted.append(lw_pair)

dict_label_to_word_final_list = {}
for item in label_word_sorted:
    dict_label_to_word_final_list[item[0]] = item[1]          
#%%
dict_counts_sub0 = {}
dict_areas_sub0 = {}
dict_captions_sub0 = {}
dict_images_sub0 = {}
for d in data0:
    print('...........................................................')
    im_path = d['image']
    im_name = im_path.split('/')[1]
    imID = img_filenames_to_img_id [im_name] 
    annId_img = coco.getAnnIds( imgIds=imID, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    objs = {}
    for annitem in anns_image:
        l = cats_id_to_name [annitem ['category_id'] ]
        a = annitem ['area']
        if l not in objs:
            objs[l] = [a]
        else:
            objs[l].append(a)
     
 
            
    cap = d['caption']['text']
    cap = cap.lower()
    print(cap)
    for obj,list_areas  in objs.items():
        names_obj = dict_label_to_word_final_list[obj]
        for n in names_obj:
            w = ' ' + n + ' '
            if w in cap:
                
                # a hit is detected
                mean_area = np.mean(list_areas)
                
                if obj not in dict_areas_sub0:
                    dict_areas_sub0[obj] = [mean_area]
                else:
                    dict_areas_sub0[obj].append(mean_area)
                
                pair = (obj, n)             
                if obj in dict_counts_sub0:
                    dict_counts_sub0[ obj] += 1
                    dict_captions_sub0 [obj].append(cap)
                    dict_images_sub0 [obj].append(im_path)
                    print(pair)
                    
                else:
                    dict_counts_sub0[ obj] = 1
                    dict_captions_sub0 [obj] = []
                    dict_captions_sub0 [obj].append(cap)
                    dict_images_sub0 [obj] = []
                    dict_images_sub0 [obj].append(im_path)
                    print(pair)
                    
dict_mean_areas_sub0 = {}                      
for key, value in dict_areas_sub0.items():
    dict_mean_areas_sub0[key] = np.mean(value)

dict_mean_areas_sub0_sorted = sorted (dict_mean_areas_sub0.items(), key=lambda x:x[1], reverse=True)    
#%%
# Khazar : az inja shoru kon, algorithm bala ro check kon, makhsoosan baraye area
#%%
images_test = images_subs
dict_images_test = {}

for im_path in images_test:
    im_name = im_path.split('/')[1]
    img = img_filenames_to_all [im_name] 

    image_id = img ['id']
    h = img ['height']
    w = img ['width']
    name = img ['file_name']
    imPath = os.path.join(dataDir, dataType, name )
    
    if im_path not in dict_images_test:
        
        im_name = im_path.split('/')[1]
        img = img_filenames_to_all [im_name] 

        image_id = img ['id']
        h = img ['height']
        w = img ['width']
        name = img ['file_name']
        imPath = os.path.join(dataDir, dataType, name )
        image = cv2.imread(imPath)
        
        annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
        anns_image = coco.loadAnns(annId_img)
        
        dict_images_test[im_path] = {}
        dict_images_test[im_path]['image_name'] = im_name
        dict_images_test[im_path]['image_id'] = image_id
        dict_images_test[im_path]['h'] = h
        dict_images_test[im_path]['w'] = w
        dict_images_test[im_path]['full_path'] = imPath
        dict_images_test[im_path]['annId_img'] = annId_img
        dict_images_test[im_path]['anns_image'] = anns_image
        cat_ids = []
        for item in anns_image:
            cat_ids.append(item['category_id'])
            
        dict_images_test[im_path]['cat_ids'] = cat_ids 
###############################################################################
                ############# masking images #############
###############################################################################

# For each category id, get images
dict_id_to_image = {}

for key, value in dict_images_test.items():
    cat_ids = list(set(value['cat_ids']))
    print(cat_ids)
    for objID in cat_ids:
        if objID not in dict_id_to_image:
            dict_id_to_image[objID] = []
            dict_id_to_image[objID].append(key)
        else:
            dict_id_to_image[objID].append(key) 
s  = 0
for key, value in dict_id_to_image.items():
    s += len(value)
    
#%% measure overlap of objects
dict_metadata_sub0 = {}
dict_obj_sorted_image = {}
dict_areas = {}
for key, value in dict_id_to_image.items():
    objID = key
    dict_obj_sorted_image[objID] = {}
    temp_image_names = []
    temp_overlaps = []
    temp_obj_area_ratio = []
    for image_candidate in value:
        
        info = dict_images_test [image_candidate]
        # index_objID = [i for i in range(len(info['cat_ids'])) if info['cat_ids'][i] == objID ]
        # index_rest = [i for i in range(len(info['cat_ids'])) if info['cat_ids'][i] != objID ]
        h = info['h']
        w = info['w']
        area = h*w
        anns_image = info['anns_image']
        mask_obj = np.zeros([h,w])
        mask_rest = np.zeros([h,w])
        for item in anns_image : # constructing true mask by ading all mask items
            mask_temp = coco.annToMask(item)
            if item['category_id'] == objID:
                mask_obj = mask_obj + mask_temp
            else:
                mask_rest = mask_rest + mask_temp
        # converting the mask to 0 and 1 binary values        
        mask_binary_obj = 1 * (mask_obj > 0 )
        mask_binary_rest = 1 * (mask_rest > 0 )
        
        area_object = sum(sum(mask_binary_obj))
        obj_area_ratio = round(area_object/area,4)
        # we add both binary masks,
        # the number of elements bigger than 1 will show the number of overlapping pixels
        mask_all = mask_binary_obj + mask_binary_rest
        mask_overlap = 1 * (mask_all > 1 )      
        overlap = sum(sum(mask_overlap))
        #print(overlap)
        #if obj_area_ratio >= 0.05:
        temp_image_names.append(image_candidate)
        temp_overlaps.append(overlap)
        temp_obj_area_ratio.append(obj_area_ratio)
    # sorting overlaps and save the image list based on sorted values    
    sorted_args = np.argsort(temp_overlaps) 
    temp_image_names_sorted = [temp_image_names[i] for i in sorted_args]
    temp_obj_area_ratio_sorted = [temp_obj_area_ratio[i] for i in sorted_args]
    dict_obj_sorted_image[objID]['areas'] = temp_obj_area_ratio_sorted
    dict_obj_sorted_image[objID]['images'] = temp_image_names_sorted
    dict_areas[objID] = np.mean(temp_obj_area_ratio_sorted)



