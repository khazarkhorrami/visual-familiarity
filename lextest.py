path_input = "/worktmp2/hxkhkh/current/lextest/outputFB/prefb/output/COCO"
txtname = 'lextest_overall_score.txt'

#%%
import os
import numpy as np
from matplotlib import pyplot as plt

def read_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

def read_all_layers (model_name):
    scores = []
    layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11']
    for layer_name in layer_names:
        name = layer_name
        path = os.path.join(path_input, model_name , name , txtname)
        s = read_score (path)
        scores.append(s)
    bs = np.sort(scores)[::-1][0]
    bl = np.argsort(scores)[::-1][0]
    return scores, bs, bl

##################################################################
                        ### vfsub3  ###
##################################################################
#%%
models = ['expS1', 'expS2', 'expS3']
dict_scores = {}
dict_bs = {}
dict_bl = {}

for model in models:
    scores, bs, bl = read_all_layers (model)
    dict_scores[model] = scores
    dict_bs[model] = bs
    dict_bl[model] = bl

#%%
x = []
results = []
for key, value in dict_bs.items():
    x.append(key)
    results.append(value)
    
n = len(x)    
barWidth = 0.25
  
br = np.arange(n)
plt.bar(br, results, color ='r', width = barWidth, edgecolor ='grey')