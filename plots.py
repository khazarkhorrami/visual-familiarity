# global variables

root = "/worktmp2/hxkhkh/current/"
dtype = 'CDI'
mtype = 'DINO'

layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11' , 'L12']
models = ['exphh','expS1', 'expS0', 'expS2', 'expS3']
names = ["6 months (SSL)", "8 months", "10 months", "10 months uniform", "12 months"]

import os
import numpy as np
from matplotlib import pyplot as plt
import json

#%% Recall
# manually enter numbers for recall@10




#%% ABX 
import csv

def read_score_abx (path):
    with open(path , 'r') as file:
      csvreader = csv.reader(file)
      data = []
      for row in csvreader:
        print(row)
        data.append(row)
        
    score = data[1][3]
    return round(100 * float(score) , 2)

def read_all_layers_abx (model_name, tn):
    scores = []
    
    for layer_name in layer_names:
        name = layer_name
        path = os.path.join(path_input, model_name , name , tn)
        s = read_score_abx (path)
        scores.append(s)
    bs = np.sort(scores)[0]
    bl = np.argsort(scores)[0]
    return scores, bs, bl

def plotbar_single_abx (names, results , title, cl,logmel):
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
    #plt.savefig(savepath + title + '.png' ,  format = 'png' )
    plt.show()

#    
path_input = os.path.join(root, 'output/WC/', mtype )
tn = 'score_phonetic.csv'
cl = 50
mfcc = 10.95


dict_scores = {}
dict_bs = {}
dict_bl = {}

for model in models:
    scores, bs, bl = read_all_layers_abx (model, tn)
    dict_scores[model] = scores
    dict_bs[model] = bs
    dict_bl[model] = bl

x = []
results = []
for key, value in dict_bs.items():
    x.append(key)
    results.append(value)


plotbar_single_abx (names, results, 'DINO', cl, mfcc)
#%%
#  Lexical score


def read_lexical_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

def read_all_layers_lexical (path, model_name, tn):
    scores = []
    layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11']
    for layer_name in layer_names:
        name = layer_name
        path_full = os.path.join(path , model_name,  name , tn)
        s = read_lexical_score (path_full)
        scores.append(s)
    bs = np.sort(scores)[::-1][0]
    bl = np.argsort(scores)[::-1][0]
    return scores, bs, bl

def get_best_lexical_score(path, models, tn):
    dict_scores = {}
    dict_bs = {}
    dict_bl = {}
    
    for model in models:
        scores, bs, bl = read_all_layers_lexical (path, model, tn)
        dict_scores[model] = scores
        dict_bs[model] = bs
        dict_bl[model] = bl
    return dict_scores, dict_bs, dict_bl

def plotbar_single_lex (names, results , title, cl,logmel):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results)
    
    br1 = np.arange(n)

    ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='black', label='log-Mel features', linewidth=1.5)
    plt.bar(br1, results, color ='b', width = barWidth,
            edgecolor ='grey')
    plt.ylabel('lexical test' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.ylim(0,1) 
    plt.legend(fontsize = 26, framealpha=0.1)
    plt.grid()
    plt.title(title + '\n', fontweight ='bold', fontsize = 26)
    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + 'lex' + title + '.png' ,  format = 'png' )
    plt.show()
    
def plotbar_multi_lex(names, results , title, cl, logmel):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results[0])
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    x = [-barWidth]
    x.extend(list(br1))
    x.extend([float (list(br1)[-1] + 3*barWidth )])
   
    ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='black', label='log-Mel features', linewidth=1.5)
    
    plt.bar(br1, results[0], color ='b', width = barWidth,
            edgecolor ='grey', label =title[0])
    plt.bar(br2, results[1], color ='g', width = barWidth,
            edgecolor ='grey', label =title[1])
    plt.bar(br3, results[2], color ='grey', width = barWidth,
            edgecolor ='grey', label =title[2])
    plt.title("comparison between all pretrained versions" + ' (' + dtype + ')' , fontweight ='bold', fontsize = 28)
    plt.ylabel('lexical test' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    #plt.ylim(0,1) 
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + 'lexall' + '_' + dtype + '_' + mtype + '.png' ,  format = 'png' )
    plt.show()
 
root_lexical = os.path.join(root, "lextest")
tn = 'output.txt'
cl = 1/89
logmel = 0.17
   
if mtype=='RCNN' or mtype=='DINO':
    models = ['expS1', 'expS0', 'expS2', 'expS3']
    names = [ "2\n months", "4\n months\n (uniform)", "4\n months","6\n months"]
elif mtype=='SSL':
    models = ['exphh']
    names = ["SSL 6M"]


exp = 'expFB'
expname = "FB"
title = mtype + '-' + dtype + '-' + expname
path_input = os.path.join(root_lexical, 'output/', dtype, mtype, exp )
dict_scores, dict_bs, dict_bl = get_best_lexical_score(path_input, models, tn)
x_FB = []
results_FB = []
for key, value in dict_bs.items():
    x_FB.append(key)
    results_FB.append(value)

plotbar_single_lex (names, results_FB, title, cl, logmel)    

#%%  semantic score

root_semantic = os.path.join(root, "semtest")

Sname_to_hname = {}
Sname_to_hname ["S1_aL_vO"] = "8 months"
Sname_to_hname ["S0_aL_vO"] = "10 months uniform"
Sname_to_hname ["S2_aL_vO"] = "10 months"
Sname_to_hname ["S3_aL_vO"] = "12 months"

file = os.path.join(root, "semtest_files_pairings.json")
with open(file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
words = []
objects = []
categories_all = []
for key, value in data.items():
    words.append(key)
    objects.append(value)
    categories_all.append(value.split('_')[0])


dict_word_to_obj_ind = {}

tt = []
categories = []
for t in range(0,1600,20):
    x = list(range(t,t+20))
    tt.append(x)
    categories.append(categories_all[t])
    
for chunk in tt:
    for element in chunk:
        dict_word_to_obj_ind [element] = chunk
        
def find_degree_per_category (category_index, S):
    chunk_rows = tt [category_index]
    d_category = []
    for row_index in chunk_rows:
        d = find_degree_per_row (row_index, S)
        d_category.append(round(d,3))
    return d_category

#### here calculates degree for each row 
def find_degree_per_row (row_index, S):
    row = S[row_index, :]
    green_window_index = dict_word_to_obj_ind [row_index]
    green_window = [row [i] for i in green_window_index]
    red_window_index = [i for i in range(0,1600) if i not in green_window_index]
    red_window = [row [i] for i in red_window_index]
    degree_row = []
    for q in green_window:
         
        z = []
        z.append(q)
        z.extend(red_window)
        z_sorted = list(np.argsort(z)) # as koochik be bozorg
        argq = z_sorted.index(0) #978
        # shifting indexs by 1 to avoid zero in denominator
        # after = 1581 -  ( argq + 2)
        # degree = after / (argq +1 ) 
        # ... new method
        # wanted = argq 
        d = argq / 1599 
        degree_row.append(round(d,3))
    return np.average(degree_row)

def measure_3 (S): 
    scores_degree_all = []
    scores_degree_cats = []
    scores_degree_cats_average = {}
    
    for category_index in range(80):
        d_cat = find_degree_per_category (category_index, S)
        scores_degree_all.extend(d_cat)
        scores_degree_cats.append(d_cat)
        scores_degree_cats_average[categories[category_index]] = round (np.average(d_cat),3)
    
    score_all = round(np.average(scores_degree_all) , 3)
    score_sorted = sorted(scores_degree_cats_average.items(), key=lambda x:x[1], reverse=True)
    print(f"Measurement 3 : {score_all}")
    print(score_sorted[0:5])    
    return score_all, score_sorted
        
def find_measure3 (S_path, Snames):  
    ss = []
    scats = []   
    for Sname in Snames:
        P = os.path.join(S_path , Sname)
        S = np.load( P + ".npy")          
        s, scat  = measure_3 (S)
        ss.append(s)
        scats.append(scat)
    return ss, scats  

def plotbar_multi_sem (names, results , title, yname, cl):
    barWidth = 0.25
    plt.subplots(figsize =(12, 12))  
    
    n = len(results[0])
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    x = [-barWidth]
    x.extend(list(br1))
    x.extend([float (list(br1)[-1] + 3*barWidth )])
   
    ychance = np.ones(len(x))*cl
    plt.plot(x,ychance, color ='red', label='chance level', linewidth=1.5) 
    
    plt.bar(br1, results[0], color ='b', width = barWidth,
            edgecolor ='grey', label ='Original image')
    plt.bar(br2, results[1], color ='g', width = barWidth,
            edgecolor ='grey', label ='Masked image')
    plt.bar(br3, results[2], color ='grey', width = barWidth,
            edgecolor ='grey', label ='Blurred image')
    plt.title(title + '\n', fontweight ='bold', fontsize = 28)
    plt.ylabel('semantic test ' + '(' + yname + ')' + '\n', fontweight ='bold',fontsize=24)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 20)
    
    #plt.ylim(0,1) 
    plt.yticks(fontsize=20)
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + title + yname + '.png' ,  format = 'png' )
    plt.show()

# individual DINO
ttype = 'expFB'
title = mtype + ', Pre' + ttype[-2:]
S_path = os.path.join(root, 'S', mtype, ttype)
#if mtype == "vfsubsets":
# Measure 3
Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
s_O, cat_O = find_measure3 (S_path ,Snames)
Snames = ["S1_aL_vM","S0_aL_vM","S2_aL_vM","S3_aL_vM"  ]
s_M, cat_M = find_measure3 (S_path ,Snames)
Snames = ["S1_aL_vB","S0_aL_vB","S2_aL_vB","S3_aL_vB"  ]
s_B, cat_B = find_measure3 (S_path ,Snames)

# plotting
names = ["8\n months", "10\n months\n (uniform)", "10\n months","12\n months"]

results = [s_O, s_M, s_B ]
plotbar_multi_sem (names, results, title, yname = 'm3', cl = 0.50)
