import os
import numpy as np
import scipy.io
from pycocotools.coco import COCO
import pylab
from matplotlib import pyplot as plt
import json

###############################################################################
                    ############# Template #############
###############################################################################
# dataDir='../data/coco_pyp/MSCOCO'
# dataType='val2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco=COCO(annFile)
# cats = coco.cats
# cat_ids = coco.getCatIds()

# cats_list = coco.loadCats(cat_ids)
# cat_names = [cat["name"] for cat in cats_list]

# image_id = int(42)
# img = coco.loadImgs(image_id)[0]
# annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
# anns_image = coco.loadAnns(annId_img)
# for item in range(len(anns_image)):
#     item_catId = anns_image[item]['category_id']
#     item_catinfo = coco.loadCats(item_catId)[0]

# query_id = cat_ids[0]
# query_annotation = coco.loadCats([query_id])[0]
# query_name = query_annotation["name"]
# query_supercategory = query_annotation["supercategory"]

# img_ids = coco.getImgIds(catIds=[query_id])


###############################################################################

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='train2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco_train, cats, cat_ids = read_data_from_path (dataDir, dataType)
# cat_names = get_cats_names (coco_train)
# cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco_train)
# supercats_ids = get_supercats_ids (cats)
# supercats_names = get_supercats_names (cats)
# img_ids_train = get_all_image_ids (coco_train)

# image_id = img_ids_train[0] # e.g., int(42)
# img, anns_image, items = read_image_anns (image_id, coco_train)
# image_anns_names = get_image_anns (items) 

# query_id = cat_ids[0] # e.g., person
# query_name, query_supercategory = read_catitem_info (query_id)
# img_ids_query = get_catitem_images (query_id)

###############################################################################
            ############# COCO main functions #############
###############################################################################



def read_data_from_path (dataDir, dataType):
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)   
    coco=COCO(annFile)
    cats = coco.cats
    cat_ids = coco.getCatIds()
    return coco, cats, cat_ids

def get_all_image_ids (coco):
    img_ids = coco.getImgIds()
    return img_ids

def get_all_cats (coco):
    cat_ids = coco.getCatIds()
    cats_list = coco.loadCats(cat_ids)
    cats_id_to_name = {}
    cats_id_to_supername = {}
    cats_name_to_id = {}
    for item in cats_list:
        cats_id_to_name [item['id']] = item['name']
        cats_id_to_supername [item['id']] = item['supercategory']
        cats_name_to_id [item['name']] = item['id']
    return cats_id_to_name, cats_id_to_supername, cats_name_to_id

def get_cats_names (coco):
    cat_ids = coco.getCatIds()
    cats_list = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats_list]
    return cat_names

def get_supercats_ids (cats):
    supercats_ids = {}
    for key, item in cats.items():
        item_id = item['id']
        item_super = item['supercategory']
        if item_super not in supercats_ids:
            supercats_ids [item_super] = []
            supercats_ids[item_super].append(item_id)
        else:
            supercats_ids[item_super].append(item_id)
    return supercats_ids

def get_supercats_names (cats):
    supercats_names = {}
    for key, item in cats.items():
        item_name = item ['name']
        item_super = item['supercategory']
        if item_super not in supercats_names:
            supercats_names [item_super] = []
            supercats_names [item_super].append(item_name)
        else:
            supercats_names[item_super].append(item_name)
    return supercats_names

def read_image_anns (image_id, coco):
    img = coco.loadImgs(image_id)[0]
    annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    items = []
    for item in range(len(anns_image)):
        item_catId = anns_image[item]['category_id']
        item_catinfo = coco.loadCats(item_catId)[0]
        items.append(item_catinfo)
    return img, anns_image, items

def get_image_anns (items):
    image_anns_names = {}
    for item in items:
        image_anns_names[ item ['name'] ] = item ['id']
    return image_anns_names

def read_catitem_info (query_id, coco):
    query_info = coco.loadCats([query_id])[0]
    query_name = query_info["name"]
    query_supercategory = query_info["supercategory"]
    return query_name, query_supercategory

def get_catitem_images (query_id, coco):
    img_ids_query = coco.getImgIds(catIds=[query_id])
    return img_ids_query
    
