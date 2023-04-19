
import os
import numpy as np
import pylab
from matplotlib import pyplot as plt
import json
from utilsMSCOCO import *




###############################################################################
                ############# COCO aux functions #############
###############################################################################

def get_images_per_cats (cat_ids, coco) :
    all_counts_images = {}
    all_images_cats = {}
    all_images_supercats = {}  
    for query_id in cat_ids:
        query_name, query_supercategory = read_catitem_info (query_id, coco)
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

def save_plot (save_path, all_counts_images,all_images_supercats ):
    save_name = save_path + '_cats'
    objects = list(all_counts_images.keys())
    values = list (all_counts_images.values())
    plot_dist_cats (objects, values, save_name)
    save_name = save_path + '_supercats'
    objects = list(all_images_supercats.keys())
    values = list(all_images_supercats.values())
    plot_dist_cats(objects, [len(item) for item in values], save_name)
    
def create_subset (objects_sorted, values_sorted, dataType, coco, k, n):
    subset = {}
    objects = objects_sorted[0:k]
    cats_id_to_name, cats_id_to_supername, cats_name_to_id = get_all_cats (coco)
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
                image_files.append(os.path.join(dataType , img_info["file_name"]))
    return subset, objects , image_files      

def creating_final_subsets (dataDir, dataType , k, n):
    
    coco, cats, cat_ids = read_data_from_path (dataDir, dataType)
    all_counts_images, all_images_cats, all_images_supercats =  get_images_per_cats (cat_ids, coco) 
     
    objects = list(all_counts_images.keys())
    values = list (all_counts_images.values())
    sorted_ind, objects_sorted, values_sorted = sort_object (objects, values)    
    subset, objects , image_files = create_subset (objects_sorted, values_sorted, dataType, coco, k, n)
    return subset, objects , image_files

def save_json_subsets (data_root, image_files_train, jname_orig, jname_sub):

    audio_dataset_json_file = os.path.join(data_root, jname_orig)
    with open(audio_dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
    data = data_json['data']
    
    data_sub = [] 
    for item in data:
        imfile = item['image']
        if imfile in image_files_train:
            data_sub.append(item)
            
    # writing json file        
    dictionary = {}
    dictionary ['data'] = data_sub
    json_object = json.dumps(dictionary, indent=4)
    json_file = os.path.join(data_root, jname_sub ) 
    # Writing to sample.json
    with open(json_file, "w") as outfile:
        outfile.write(json_object)
    
    return data_sub

#############################################
# plotting the distribution of the objects
#############################################

######################### reading train and val data

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='train2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco_train, cats, cat_ids = read_data_from_path (dataDir, dataType)
# all_counts_images_train, all_images_cats_train, all_images_supercats_train =  get_images_per_cats (cat_ids, coco_train) 

# dataDir='../data/coco_pyp/MSCOCO'
# dataType='val2014'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# coco_val, cats, cat_ids = read_data_from_path (dataDir, dataType)
# all_counts_images_val, all_images_cats_val, all_images_supercats_val =  get_images_per_cats (cat_ids, coco_val) 

# ######################## unifying train and val sets

# all_counts_images = {}
# all_images_cats = {}
# all_images_supercats = {}
# for key in all_counts_images_train:
#     all_counts_images [key] = all_counts_images_train [key] + all_counts_images_val [key]
#     all_images_cats [key] = all_images_cats_train [key] + all_images_cats_val [key]
# for key in all_images_supercats_train:
#     all_images_supercats [key] = all_images_supercats_train [key] + all_images_supercats_val [key]

# ######################## plotting 

# save_path = '/worktmp2/hxkhkh/current/FaST/experiments/plots/vf/distributions/'
# save_plot (save_path + 'all', all_counts_images, all_images_supercats )
# save_plot (save_path + 'train', all_counts_images_train, all_images_supercats_train )
# save_plot (save_path + 'val', all_counts_images_val, all_images_supercats_val )

###############################################################################
                ############# creating subsets #############
###############################################################################

######################## checking if in Karpathy train split, there is any val file

# data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"
# audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")

# with open(audio_dataset_json_file, 'r') as fp:
#     data_json = json.load(fp)
    
# data = data_json['data']
# count_train = 0
# for item in data:
#     namef = item['image'].split('/')[0]
#     if 'train' in namef:
#         count_train += 1
        
# Results is zero; there is no val file in Karpathy train split
# So, we select out subset only from COCO train set, sinc the size is enough

######################### creating train subsets #############################

######################### reading train data

dataDir='../data/coco_pyp/MSCOCO'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
# subset 1: small subset (10 most frequent objects)
##################################################
k1 = 10 
n1 = 150 # how many images for each object
subset_train_1, objects_train_1 , image_files_train_1 = creating_final_subsets (dataDir, dataType , k1, n1)
# subset 2: medium subset (40 most frequent objects)
###################################################
k2 = 40 
n2 = 150 # how many images for each object
subset_train_2, objects_train_2 , image_files_train_2 = creating_final_subsets (dataDir, dataType , k2, n2)


######################### creating val subsets #############################

######################### reading val data

dataDir='../data/coco_pyp/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    
# subset 1: small subset (10 most frequent objects)
##################################################
k1 = 10 
n1 = -1 # how many images for each object
subset_val_1, objects_val_1 , image_files_val_1 = creating_final_subsets (dataDir, dataType , k1, n1)
# subset 2: medium subset (40 most frequent objects)
###################################################
k2 = 40 
n2 = -1 # how many images for each object
subset_val_2, objects_val_2 , image_files_val_2 = creating_final_subsets (dataDir, dataType , k2, n2)

###############################################################################
                ############# saving subsets #############
###############################################################################
 
data_root = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp"

# for train split

jname_orig = "SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json"

jname_sub = "SpokenCOCO/SpokenCOCO_train_sub1_unrolled_karpathy.json"
data_sub_train_1 = save_json_subsets (data_root, image_files_train_1, jname_orig, jname_sub)

jname_sub = "SpokenCOCO/SpokenCOCO_train_sub2_unrolled_karpathy.json"
data_sub_train_2 = save_json_subsets (data_root, image_files_train_2, jname_orig, jname_sub)

# for val split 

jname_orig = "SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json"

jname_sub = "SpokenCOCO/SpokenCOCO_val_sub1_unrolled_karpathy.json"
data_sub_val_1 = save_json_subsets (data_root, image_files_val_1, jname_orig, jname_sub)

jname_sub = "SpokenCOCO/SpokenCOCO_val_sub2_unrolled_karpathy.json"
data_sub_val_2 = save_json_subsets (data_root, image_files_val_2, jname_orig, jname_sub)
