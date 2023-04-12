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

dataDir='../data/coco_pyp/MSCOCO'

dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)



coco=COCO(annFile)
cats = coco.cats
cat_ids = coco.getCatIds()

cats_list = coco.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats_list]

image_id = int(42)
img = coco.loadImgs(image_id)[0]
annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
anns_image = coco.loadAnns(annId_img)
for item in range(len(anns_image)):
    item_catId = anns_image[item]['category_id']
    item_catinfo = coco.loadCats(item_catId)[0]

query_id = cat_ids[0]
query_annotation = coco.loadCats([query_id])[0]
query_name = query_annotation["name"]
query_supercategory = query_annotation["supercategory"]

img_ids = coco.getImgIds(catIds=[query_id])

all_objects = {}
all_images = {}
all_ids = {}
for query_id in cat_ids:
    img_ids = coco.getImgIds(catIds=[query_id])
    query_annotation = coco.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    all_objects[query_name] = len(img_ids)
    all_images[query_name] = img_ids
    all_ids[query_name] = query_id
        
# sort dictionary by values   
sortedobjs = sorted(all_objects.items(), key=lambda x:x[1], reverse=True)

for t in sortedobjs:
    obj, count = t
    img_ids = coco.getImgIds(catIds=[query_id])
    
#############################################
# plotting the distribution of the objects
#############################################

objects = list(all_objects.keys())
values = list (all_objects.values())

sorted_ind = np.argsort(values)[::-1]
objects_sorted = [objects[i] for i in sorted_ind ]
values_sorted = [values[j] for j in sorted_ind]


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

#############################################
# subset 1: small subset (10 most frequent objects)
#############################################
subset_1 = {}
objects_1 = objects_sorted[0:10]
values_1 = values_sorted[0:10]
images_1 = []
image_files_1 = []
# how many images for each object
n = 150
for obj in objects_1:
    objID = all_ids[obj]
    img_ids = coco.getImgIds(catIds=[objID])[0:n]
    subset_1[obj] = {}
    subset_1[obj]['objID'] = objID
    subset_1[obj]['images'] = img_ids
    for item in img_ids [0:n]:
        if item not in images_1:
            images_1.append(item)
            img_info = coco.loadImgs([item])[0]
            image_files_1.append(img_info["file_name"])
            

#############################################
# subset 2: medium subset (40 most frequent objects)
#############################################
subset_2 = {}
objects_2 = objects_sorted[0:40]
values_2 = values_sorted[0:40]
images_2 = []
image_files_2 = []
# how many images for each object
n = 150
for obj in objects_2:
    objID = all_ids[obj]
    img_ids = coco.getImgIds(catIds=[objID])[0:n]
    subset_2[obj] = {}
    subset_2[obj]['objID'] = objID
    subset_2[obj]['images'] = img_ids
    for item in img_ids [0:n]:
        if item not in images_2:
            images_2.append(item)
            img_info = coco.loadImgs([item])[0]
            image_files_2.append(img_info["file_name"])

