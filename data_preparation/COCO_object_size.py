import numpy as np
import cv2
import os
import json 
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids, change_labels
root = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/"
jsub0 = os.path.join(root, "SpokenCOCO_train_subset0A.json")
# jsub1 = os.path.join(root, "SpokenCOCO_train_subset1.json")
# jsub2 = os.path.join(root, "SpokenCOCO_train_subset2.json")
# jsub3 = os.path.join(root, "SpokenCOCO_train_subset3.json")
#%%

def read_images_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    images = []
    for d in data:
        im = d['image']      
        images.append(im)
    return images

imsub0 = read_images_json(jsub0)
# imsub1 = read_images_json(jsub1)
# imsub2 = read_images_json(jsub2)
# imsub3 = read_images_json(jsub3)
#%%
saveDir = "../../../semtest/images"
dataDir='../../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco , cats, cat_ids = read_data_from_path (dataDir, dataType)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)

img_ids = get_all_image_ids (coco)

img_id_to_filenames = {}
img_filenames_to_id = {}
img_filenames_to_all = {}
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    fn = img['file_name']
    iD = img['id']
    img_id_to_filenames[iD] = fn
    img_filenames_to_id [fn] = iD
    img_filenames_to_all [fn] = img

all_labels, cats_id_to_short_name = change_labels (cats_id_to_name)  

#%%
caption_json = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected.json"
with open(caption_json, 'r') as fp:
    dict_words_selected = json.load(fp)
#%%

images_test = imsub0
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
kh
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



