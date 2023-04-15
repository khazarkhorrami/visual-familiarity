#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:31:49 2023

@author: hxkhkh
"""

import os
import numpy as np
import scipy.io
from pycocotools.coco import COCO
import pylab
from matplotlib import pyplot as plt

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

def read_catitem_info (query_id):
    query_info = coco.loadCats([query_id])[0]
    query_name = query_info["name"]
    query_supercategory = query_info["supercategory"]
    return query_name, query_supercategory

def get_catitem_images (query_id):
    img_ids_query = coco.getImgIds(catIds=[query_id])
    return img_ids_query
    
###############################################################################

dataDir='../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco, cats, cat_ids = read_data_from_path (dataDir, dataType)
cat_names = get_cats_names (coco)
cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)
supercats_ids = get_supercats_ids (cats)
supercats_names = get_supercats_names (cats)
img_ids = get_all_image_ids (coco)


image_id = img_ids[0] # e.g., int(42)
img, anns_image, items = read_image_anns (image_id, coco)
image_anns_names = get_image_anns (items) 

query_id = cat_ids[0] # e.g., person
query_name, query_supercategory = read_catitem_info (query_id)
img_ids_query = get_catitem_images (query_id)

###############################################################################
                ############# COCO aux functions #############
###############################################################################

def get_images_per_cats (cat_ids) :
    all_counts_images = {}
    all_images_cats = {}
    all_images_supercats = {}  
    for query_id in cat_ids:
        query_name, query_supercategory = read_catitem_info (query_id)
        img_ids_query = coco.getImgIds(catIds=[query_id])
        all_counts_images[query_name] = len(img_ids_query)
        all_images_cats [query_name] = img_ids_query
        if query_supercategory not in all_images_supercats:
            all_images_supercats [query_supercategory] = []
            all_images_supercats [query_supercategory].extend(img_ids_query)
        else:
            all_images_supercats [query_supercategory].extend(img_ids_query)
    return all_counts_images, all_images_cats, all_images_supercats
    

def sort_object (objects, values):
    sorted_ind = np.argsort(values)[::-1]
    objects_sorted = [objects[i] for i in sorted_ind ]
    values_sorted = [values[j] for j in sorted_ind]  
    return sorted_ind, objects_sorted,values_sorted 

def plot_dist_cats (objects, values, save_name):
    sorted_ind, objects_sorted,values_sorted = sort_object (objects, values)
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
    ax.set_title('Number of images for annotated objects',
                 loc ='center', )
    if save_name:
        plt.savefig(save_name, format='pdf')

def create_subset (objects_sorted, values_sorted, k, n):
    subset = {}
    objects = objects_sorted[0:k]
    #values_1 = values_sorted[0:10]
    images = []
    image_files = []     
    for obj in objects:
        objID = cats_name_to_id[obj]
        img_ids_query = coco.getImgIds(catIds=[objID])[0:n]
        subset[obj] = {}
        subset[obj]['objID'] = objID
        subset[obj]['images'] = img_ids_query
        for item in img_ids_query [0:n]:
            if item not in images:
                images.append(item)
                img_info = coco.loadImgs([item])[0]
                image_files.append(img_info["file_name"])
    return subset, objects , image_files      

#############################################
# plotting the distribution of the objects
#############################################

# all_counts_images, all_images_cats, all_images_supercats =  get_images_per_cats (cat_ids) 
# save_path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/'
# save_name = save_path + 'distribution_cats'
# objects = list(all_counts_images.keys())
# values = list (all_counts_images.values())
# plot_dist_cats (objects, values, save_name)
# save_name = save_path + 'distribution_supercats'
# objects = list(all_images_supercats.keys())
# values = list(all_images_supercats.values())
# plot_dist_cats(objects, [len(item) for item in values], save_name)

###############################################################################
                ############# creating subsets #############
###############################################################################
all_counts_images, all_images_cats, all_images_supercats =  get_images_per_cats (cat_ids) 


objects = list(all_counts_images.keys())
values = list (all_counts_images.values())
sorted_ind, objects_sorted, values_sorted = sort_object (objects, values)


# subset 1: small subset (10 most frequent objects)
##################################################
k = 10 
n = 150 # how many images for each object
subset_1, objects_1 , image_files_1 = create_subset (objects_sorted, values_sorted, k, n)

# subset 2: medium subset (40 most frequent objects)
###################################################
k = 40 
n = 150 # how many images for each object
sorted_ind, objects_sorted, values_sorted = sort_object (objects, values)
subset_2, objects_2 , image_files_2 = create_subset (objects_sorted, values_sorted, k, n)




