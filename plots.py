
import os
import numpy as np
from matplotlib import pyplot as plt
import json
import csv

# global variables
root = "/worktmp2/hxkhkh/current/"
path_abx = os.path.join(root, 'ZeroSpeech/output/WC/vfsubsets')
path_lex = os.path.join(root, "lextest/output/CDI/DINO/expFB")
path_sem = os.path.join(root, "semtest/S/DINO/expFB")
path_save = "/worktmp2/hxkhkh/current/FaST/papers/vf/material/"
dtype = 'CDI'
mtype = 'DINO'

layer_names = ['L0','L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11' ]
models = ['exphh','expS1', 'expS2', 'expS0', 'expS3']
names = ["6 M\n(speech)", "8 M", "10 M", "10 M\n (uniform)", "12 M"]
Sname_to_hname = {}
Sname_to_hname ["S1_aL_vO"] = "8 months"
Sname_to_hname ["S0_aL_vO"] = "10 months uniform"
Sname_to_hname ["S2_aL_vO"] = "10 months"
Sname_to_hname ["S3_aL_vO"] = "12 months"

# functions
def plot_all (names , results_recall, results_abx, results_lex, results_sem):
    recall_cl = 0.2
    abx_cl = 50
    abx_mfcc = 10.95
    lex_cl = 1/89
    lex_logmel = 0.17
    plt.figure(figsize =(24,24))
    f_leg = 26
    f_ticks = 26
    f_ylabel = 30
    ######### recall
    plt.subplot(2,2,1)
    [r_image, r_speech] = results_recall
    barWidth = 0.25    
    n = len(r_image)  
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    x = []
    x.extend(list(br1))
    x.extend([br2[-1]]) 
    ychance = np.ones(len(x))*recall_cl
    plt.plot(x,ychance, color ='purple', label='chance level', linewidth=5) 
    plt.bar(br1, r_image, color ='olive', width = barWidth,
            edgecolor ='black', label ="speech_to_image")
    plt.bar(br2, r_speech, color ='darkolivegreen', width = barWidth,
            edgecolor ='black', label ="image_to_speech")
    plt.ylabel('recall@10' + '\n', fontweight ='bold', fontsize = f_ylabel)
    plt.ylim(0,12)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.legend(fontsize = f_leg)
    plt.grid()
    
    ######### sem
    plt.subplot(2,2,2)
    barWidth = 0.25  
    n = len(results_sem[0])   
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    x = []
    x.extend(list(br1))
    x.extend([br2[-1]])   
    ychance = np.ones(len(x))*0.5 # chance level
    plt.plot(x,ychance, color ='purple', label='chance level', linewidth=5)    
    plt.bar(br1, results_sem[0], color ='olive', width = barWidth,
           edgecolor ='black', label ='Original image')
    plt.bar(br2, results_sem[1], color ='darkolivegreen', width = barWidth,
            edgecolor ='black', label ='Masked image')
    plt.ylabel('semantic test ', fontweight ='bold',fontsize=f_ylabel)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = f_ticks) 
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1) 
    plt.legend(fontsize = f_leg)
    plt.grid()
    
    ######### ABX
    plt.subplot(2,2,3)
    barWidth = 0.25
    n = len(results_abx)
    br1 = np.arange(n)
    ylogmel = np.ones(len(br1))*abx_mfcc
    plt.plot(br1,ylogmel, color ='blue', label='MFCC features', linewidth=5)
    plt.bar(br1, results_abx, color ='darkolivegreen', width = barWidth,
            edgecolor ='black',label='WC')
    plt.ylabel('ABX error' + '\n', fontweight ='bold', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,12) 
    plt.legend(fontsize = 20)
    plt.grid()
    plt.legend(fontsize = f_leg)
    savepath = os.path.join(root, "results/" )
    
    ######### lEX
    plt.subplot(2,2,4)
    barWidth = 0.25
    n = len(results_lex)  
    br1 = np.arange(n)
    ychance = np.ones(len(br1))*lex_cl
    ylogmel = np.ones(len(br1))*lex_logmel
    plt.plot(br1,ychance, color ='purple', label='chance level', linewidth=5)
    plt.plot(br1,ylogmel, color ='blue', label='log-Mel features', linewidth=5)
    plt.bar(br1, results_lex, color ='darkolivegreen', width = barWidth,
            edgecolor ='black')
    plt.ylabel('lexical test' , fontweight ='bold', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1) 
    plt.legend(fontsize = f_leg)#, framealpha=0.1)
    plt.grid()
    

    
    
    plt.savefig(path_save +  'results.png' ,  format = 'png' )
    
    
def plot_recall(names, results_recall , r_ch, ax):
    [r_image, r_speech] = results_recall
    barWidth = 0.25
    
    n = len(r_image)
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    #br3 = [x + barWidth for x in br2]
    x = [-barWidth]
    x.extend(list(br1))
    x.extend([float (list(br1)[-1] + 3*barWidth )])
    
    ychance = np.ones(len(br1))*r_ch
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    
    plt.bar(br1, r_image, color ='b', width = barWidth,
            edgecolor ='grey', label ="speech_to_image")
    plt.bar(br2, r_speech, color ='g', width = barWidth,
            edgecolor ='grey', label ="image_to_speech")
    
    #plt.title("recall@10" + ' (' + dtype + ')' , fontweight ='bold', fontsize = 28)
    plt.ylabel('recall@10' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    #plt.ylim(0,1) 
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + 'lexall' + '_' + dtype + '_' + mtype + '.png' ,  format = 'png' )
    

def plot_abx (names, results_abx , cl,logmel):
    barWidth = 0.25
    #fig = plt.subplots(figsize =(12, 12))  
    #plt.figure(figsize =(14, 14))
    n = len(results_abx)
    
    br1 = np.arange(n)

    #ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    #plt.plot(br1,ychance, color ='orange', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='green', label='MFCC features', linewidth=1.5)
    plt.bar(br1, results_abx, color ='b', width = barWidth,
            edgecolor ='grey',label='WC')
    plt.ylabel('ABX error' + '\n', fontweight ='bold', fontsize = 24)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.ylim(0,12) 
    plt.legend(fontsize = 20)
    plt.grid()
    plt.yticks(fontsize = 26)
    #plt.ylim(0,1) 
    plt.legend(fontsize = 24)

    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + title + '.png' ,  format = 'png' )
    plt.show()
    
def plotbar_single_lex (names, results_lex , cl,logmel):
    barWidth = 0.25
    #plt.figure(figsize =(14, 14))
    n = len(results_lex)
    
    br1 = np.arange(n)

    ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='green', label='log-Mel features', linewidth=1.5)
    plt.bar(br1, results_lex, color ='b', width = barWidth,
            edgecolor ='grey')
    plt.ylabel('lexical test' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.ylim(0,1) 
    plt.legend(fontsize = 26, framealpha=0.1)
    plt.grid()
    #plt.title(title + '\n', fontweight ='bold', fontsize = 26)
    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + 'lex' + title + '.png' ,  format = 'png' )
    plt.show()

def plotbar_sem (names, results_sem , cl):
    barWidth = 0.25
    
    n = len(results_sem[0])
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    #br3 = [x + barWidth for x in br2]
    x = [-barWidth]
    x.extend(list(br1))
    x.extend([float (list(br1)[-1] + 3*barWidth )])
   
    ychance = np.ones(len(x))*cl
    plt.plot(x,ychance, color ='red', label='chance level', linewidth=1.5) 
    
    plt.bar(br1, results_sem[0], color ='b', width = barWidth,
            edgecolor ='grey', label ='Original image')
    plt.bar(br2, results_sem[1], color ='g', width = barWidth,
            edgecolor ='grey', label ='Masked image')
    # plt.bar(br3, results[2], color ='grey', width = barWidth,
    #         edgecolor ='grey', label ='Blurred image')
    
    plt.ylabel('semantic test ', fontweight ='bold',fontsize=24)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 20)
    
    #plt.ylim(0,1) 
    plt.yticks(fontsize=20)
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    #plt.savefig(savepath + title + yname + '.png' ,  format = 'png' )
    plt.show()
############################################################################# ABX

def read_score_abx (path):
    with open(path , 'r') as file:
      csvreader = csv.reader(file)
      data = []
      for row in csvreader:
        print(row)
        data.append(row)
        
    score = data[1][3]
    return round(100 * float(score) , 2)

def read_all_layers_abx (model_name):
    scores = []
    
    for layer_name in layer_names:
        name = layer_name
        path = os.path.join(path_abx, model_name , name , 'score_phonetic.csv')
        s = read_score_abx (path)
        scores.append(s)
    bs = np.sort(scores)[0]
    bl = np.argsort(scores)[0]
    return scores, bs, bl
 
########################################################################### LEX
def read_lexical_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

def read_all_layers_lexical (path, model_name, tn):
    scores = []
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


########################################################################### SEM
file = os.path.join(root, "semtest", "semtest_files_pairings.json")
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

#%%
#%% Recall
# manually enter numbers for recall@10 [ssl, r1, r2, r0, r3]

r_image = [0.2, 0.7, 4.4, 5.0 , 9.2 ]
r_speech = [0.2, 1.1, 5.1, 5.5, 11.6 ]
results_recall = [r_image, r_speech]

# ABX 

dict_scores = {}
dict_bs = {}
dict_bl = {}
for model in models:
    scores, bs, bl = read_all_layers_abx (model)
    dict_scores[model] = scores
    dict_bs[model] = bs
    dict_bl[model] = bl
x = []
results_abx = []
for key, value in dict_bs.items():
    x.append(key)
    results_abx.append(value)

#  Lex

dict_scores, dict_bs, dict_bl = get_best_lexical_score(path_lex, models, 'output.txt')
x_FB = []
results_lex = []
for key, value in dict_bs.items():
    x_FB.append(key)
    results_lex.append(value)
  

#  SEM
# Measure 3
Snames = ["S1_aL_vO","S2_aL_vO", "S0_aL_vO","S3_aL_vO"  ]
s_O, cat_O = find_measure3 (path_sem ,Snames)
# Snames = ["S1_aL_vM","S0_aL_vM","S2_aL_vM","S3_aL_vM"  ]
# s_M, cat_M = find_measure3 (S_path ,Snames)
Snames = ["S1_aL_vB","S2_aL_vB","S0_aL_vB","S3_aL_vB"  ]
s_B, cat_B = find_measure3 (path_sem ,Snames)

# for SSL (later compute this)
s_O.insert(0, 0.5)
s_B.insert(0, 0.5)
results_sem = [s_O, s_B]
#%%

# plotting
plot_all (names , results_recall, results_abx, results_lex, [s_O, s_B])