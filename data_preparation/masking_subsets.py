import numpy
import cv2
import os
from matplotlib import pyplot as plt
from utilsMSCOCO import read_data_from_path, get_all_cats, get_all_image_ids
import copy


###############################################################################
                ############# masking images #############
###############################################################################

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
   
#%%

path_sub1 = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_subset1.json"
path_sub2 = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_subset2.json"
path_sub3 = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_subset3.json"
path_sub0 = "/worktmp2/hxkhkh/current/FaST/datavf/coco/subsets/SpokenCOCO_train_subset0A.json"

import json  
def read_wav_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_json = json.load(json_file) 
    data = data_json['data']
    images = []
    for d in data:
        w = d['image']
        images.append(w)
    return images

file_path = path_sub1
images_subset = read_wav_json(file_path)


#%%
# example
im_path = images_subset[25]
im_name = im_path.split('/')[1]
img = img_filenames_to_all [im_name] 

image_id = img ['id']
h = img ['height']
w = img ['width']
name = img ['file_name']
imPath = os.path.join(dataDir, dataType,name )
image = cv2.imread(imPath)

################################## getting mask from annotation
annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
anns_image = coco.loadAnns(annId_img)
mask_annitem = numpy.zeros([h,w])
for item in anns_image : # constructing true mask by ading all mask items
    mask_temp = coco.annToMask(item)
    mask_annitem = mask_annitem + mask_temp
    #plt.imshow(255 * mask_temp)
mask_binary = 1 * (mask_annitem > 0 )
mask_binary_uint = numpy.array(mask_binary, dtype=numpy.uint8)
mask = 1 * mask_binary_uint
plt.imshow(mask)
print (len(annId_img))

################################## masking the image
masked_image = copy.deepcopy (image)
masked_image[:,:,0] = image[:,:,0] * mask
masked_image[:,:,1] = image[:,:,1] * mask
masked_image[:,:,2] = image[:,:,2] * mask

blured_image = cv2.blur(image,(30,30),0)
masked_image_blur = copy.deepcopy (blured_image)
masked_image_blur[:,:,0] = blured_image[:,:,0] * (1- mask)
masked_image_blur[:,:,1] = blured_image[:,:,1] * (1- mask)
masked_image_blur[:,:,2] = blured_image[:,:,2] * (1- mask)
blurmasked_image = masked_image_blur + masked_image

######################### plotting the results
plt.subplot(2,2,1)
plt.imshow(image)
plt.subplot(2,2,2)
plt.imshow(mask)
plt.subplot(2,2,3)
plt.imshow(masked_image)
plt.subplot(2,2,4)
plt.imshow(blurmasked_image)

######################### saving the results
final_frame = cv2.hconcat((image, masked_image, blurmasked_image ))
# cv2.imshow('lena', final_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
savePath = '/worktmp2/hxkhkh/current/FaST/plots/vf/masked_images/examples/'
cv2.imwrite(savePath + name , final_frame)

#%%