root = "/worktmp2/hxkhkh/current/ZeroSpeech/"
mtype = 'vfsubsets'

if mtype == 'vfplus':
    title = 'RCNN' 
elif mtype == 'vfsubsets':
    title = 'DINO'


#%%
import os
import numpy as np
from matplotlib import pyplot as plt
import os

import csv

def read_score (path):
    with open(path , 'r') as file:
      csvreader = csv.reader(file)
      data = []
      for row in csvreader:
        print(row)
        data.append(row)
        
    score = data[1][3]
    return round(100 * float(score) , 2)

def read_all_layers (model_name, tn):
    scores = []
    layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11']
    for layer_name in layer_names:
        name = layer_name
        path = os.path.join(path_input, model_name , name , tn)
        s = read_score (path)
        scores.append(s)
    bs = np.sort(scores)[0]
    bl = np.argsort(scores)[0]
    return scores, bs, bl
###############################################################################

def plotbar_single (names, results , title, cl,logmel):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results)
    
    br1 = np.arange(n)

    #ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    #plt.plot(br1,ychance, color ='orange', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='green', label='MFCC features', linewidth=1.5)
    plt.bar(br1, results, color ='b', width = barWidth,
            edgecolor ='grey')
    plt.ylabel('ABX error' + '\n', fontweight ='bold', fontsize = 24)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 20)
    plt.ylim(0,12) 
    plt.legend(fontsize = 20)
    plt.grid()
    plt.title(title + '\n', fontweight ='bold', fontsize = 26)
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + title + '.png' ,  format = 'png' )
    plt.show()
#%%    
path_input = os.path.join(root, 'output/WC/', mtype )
tn = 'score_phonetic.csv'
cl = 50
mfcc = 10.95

if mtype=='vfsubsets' or mtype=='vfplus':
    models = ['exphh','expS1', 'expS0', 'expS2', 'expS3']
    names = ["Speech only\n(6 months)", "subset 1\n(2 months)", "subset 0\n(uniform)", "subset 2\n(4 months)","subset 3\n(6 months)"]
elif mtype=='vfls':
    models = ['exphh']
    names = ["SSL 6M"]
    
#%%

dict_scores = {}
dict_bs = {}
dict_bl = {}

for model in models:
    scores, bs, bl = read_all_layers (model, tn)
    dict_scores[model] = scores
    dict_bs[model] = bs
    dict_bl[model] = bl

#%%
x = []
results = []
for key, value in dict_bs.items():
    x.append(key)
    results.append(value)


plotbar_single (names, results, title, cl, mfcc)