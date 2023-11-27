
import os
import numpy as np
from matplotlib import pyplot as plt
import json
import csv

# global variables
root = "/worktmp2/hxkhkh/current/"
path_abx = os.path.join(root, 'ZeroSpeech/output/WC/vfsubsets/expFB')
path_lex = os.path.join(root, "lextest/output/CDI/DINO/exp6M")
path_sem = os.path.join(root, "semtest/S/DINO/exp6M")
path_save = "/worktmp2/hxkhkh/current/FaST/papers/vf/material/"
dtype = 'CDI'
mtype = 'DINO'

layer_names = ['L0','L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11' ]

names = ["6 ", "8 ", "10 ", "12 ", "10(u)"]
Sname_to_hname = {}
Sname_to_hname ["S1_aL_vO"] = "8 months"
Sname_to_hname ["S2_aL_vO"] = "10 months"
Sname_to_hname ["S3_aL_vO"] = "12 months"
Sname_to_hname ["S0_aL_vO"] = "10 months uniform"


# functions
def plot_all (names , results_recall, results_abx, results_lex, results_sem):
    recall_cl = 0.2
    abx_cl = 50
    abx_mfcc = 10.95
    lex_cl = 1/89
    lex_logmel = 0.17
    fig = plt.figure(figsize =(32,24))
    f_leg = 28
    f_ticks = 26
    f_ylabel = 34
    
    ######### recall
    plt.subplot(2,3,2)
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
    bars1 = plt.bar(br1, r_image, color ='olive', width = barWidth, align='center', ecolor='black', capsize=10,
            edgecolor ='black', label ="speech_to_image", alpha=0.8)

    bars2 = plt.bar(br2, r_speech, color ='brown', width = barWidth, align='center', ecolor='black', capsize=10,
            edgecolor ='black', label ="image_to_speech", alpha=0.8)#'darkolivegreen'
    plt.ylabel('cross-modal retrieval (%)', fontweight ='bold', fontsize = f_ylabel)
    plt.xlabel('\n Age (month)\n', fontsize = f_ylabel)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,12)
    
    plt.legend(fontsize = f_leg, framealpha=0.1)
    plt.grid()
    bars1[0].set_hatch('//')
    bars1[0].set_linewidth(4)
    bars2[0].set_hatch('//')
    bars2[0].set_linewidth(4)
    
    ######### sem
    [(s_O, std_O), (s_B, std_B)] = results_sem
    plt.subplot(2,3,4)
    barWidth = 0.25  
    n = len(s_O)   
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    x = []
    x.extend(list(br1))
    x.extend([br2[-1]])   
    ychance = np.ones(len(x))*0.5 # chance level
    plt.plot(x,ychance, color ='purple', label='chance level', linewidth=5)    
    bars1 = plt.bar(br1, s_O, yerr=std_O , color ='olive', width = barWidth,
           edgecolor ='black',  label ='Original image', align='center', alpha=0.8, ecolor='black', capsize=10)
    
    bars2 = plt.bar(br2, s_B,  yerr=std_B ,color ='brown', width = barWidth,
            edgecolor ='black', label ='Masked image', align='center', alpha=0.8, ecolor='black', capsize=10) #'darkolivegreen'
    plt.errorbar(br1, s_O, std_O, fmt='.', color='Black')
    plt.ylabel('Semantic score (%)', fontweight ='bold',fontsize=f_ylabel)
    plt.xlabel('\n Age (month)', fontsize = f_ylabel)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = f_ticks) 
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1.1) 
    plt.legend(fontsize = f_leg, framealpha=0.1)
    plt.grid()
    bars1[0].set_hatch('//')
    bars1[0].set_linewidth(4)
    bars2[0].set_hatch('//')
    bars2[0].set_linewidth(4)
    
    ######### lEX
    (scores_lex, std_lex) =  results_lex
    plt.subplot(2,3,5)
    barWidth = 0.25
    n = len(scores_lex)  
    br1 = np.arange(n)
    ychance = np.ones(len(br1))*lex_cl
    ylogmel = np.ones(len(br1))*lex_logmel
    plt.plot(br1,ychance, color ='purple', label='chance level', linewidth=5)
    plt.plot(br1,ylogmel, color ='blue', label='log-Mel features', linewidth=5)
    bars = plt.bar(br1, scores_lex,  yerr=std_lex , color ='olive', width = barWidth,
            edgecolor ='black', alpha=0.8 , align='center', ecolor='black', capsize=10)
    
    plt.ylabel('Lexical score (%)' , fontweight ='bold', fontsize = f_ylabel)
    plt.xlabel('\n Age (month)\n', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1.1) 
    plt.legend(fontsize = f_leg, framealpha=0.1)#, framealpha=0.1)
    plt.grid()
    bars[0].set_hatch('//')
    bars[0].set_linewidth(4)

    ######### ABX
    plt.subplot(2,3,6)
    barWidth = 0.25
    n = len(results_abx)
    br1 = np.arange(n)
    ylogmel = np.ones(len(br1))*abx_mfcc
    plt.plot(br1,ylogmel, color ='blue', label='MFCC features', linewidth=5)
    bars = plt.bar(br1, results_abx, color ='olive', width = barWidth, edgecolor='black',
           label='WC', ecolor='black', capsize=10, alpha=0.8)
    
    plt.ylabel('Phonemic error rate (%)', fontweight ='bold', fontsize = f_ylabel)
    plt.xlabel('\n Age (month)', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,15) 
    plt.legend(fontsize = f_leg, loc=1, framealpha=0.1)
    plt.grid()    
    bars[0].set_hatch('//')
    bars[0].set_linewidth(4)
    
    #legend plot
    n = 5
    br1 = np.arange(n)
    legend_ax = plt.subplot(2, 3, 1)
    legend_ax.set_axis_off()  
    bars = plt.bar(br1, br1*0, edgecolor='black', facecolor='none', hatch='//',linewidth = 4, label='Speech only learning', alpha=0.8)
    bars = plt.bar(br1, br1*0, edgecolor='black', facecolor='gray', hatch='', label='Audiovisual learning', alpha=0.3)
    legend_ax.legend(fontsize = f_leg *1.3 ,framealpha=0.1, loc='center left')
    plt.ylim(-0.1,-1) 
    
    
    #training curves
    
    ax = plt.subplot(2,3,3)
    m = np.arange(5)
    ax.plot(m, -2*m , label= "training curves come here")
    ax.legend(fontsize = f_leg, framealpha=0.1)
    
    plt.tight_layout(pad=6.0)
    plt.savefig(path_save +  'results.png' , bbox_inches='tight',  format = 'png' )
    
    
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
def read_lexical_score_classes (path):
    words= []
    scores_classes = []
    with open(path) as file:
        for line in file:
            l = line.rstrip()
            word = l.split(',')[0].strip()
            score = float(l.split(',')[1])
            words.append(word)
            scores_classes.append(score)
    return words,  scores_classes

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
    path_out_classes = os.path.join(path , model_name, 'L' + str(bl), 'output_words.txt')
    words, scores_classes = read_lexical_score_classes (path_out_classes)
    std_classes = np.std(scores_classes)
    return scores, bs, bl, std_classes

def get_best_lexical_score(path, models, tn):
    dict_scores = {}
    dict_bs = {}
    dict_bl = {}
    dict_std = {}
    
    for model in models:
        scores, bs, bl, std_classes = read_all_layers_lexical (path, model, tn)
        dict_scores[model] = scores
        dict_bs[model] = bs
        dict_bl[model] = bl
        dict_std [model] = std_classes
    return dict_scores, dict_bs, dict_bl, dict_std


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
    
    score_mean = round(np.average(scores_degree_all) , 3)
    score_std = round(np.std(scores_degree_all) , 3)
    
    score_sorted = sorted(scores_degree_cats_average.items(), key=lambda x:x[1], reverse=True)
    print(f"Measurement 3 : {score_mean}")
    print(score_sorted[0:5])    
    return score_mean, score_std, score_sorted
        
def find_measure3 (S_path, Snames):  
    ss = []
    ss_std = []
    scats = []   
    for Sname in Snames:
        P = os.path.join(S_path , Sname)
        S = np.load( P + ".npy")          
        s, s_std, scat = measure_3 (S)
        ss.append(s)
        ss_std.append(s_std)
        scats.append(scat)
    return ss,ss_std, scats  

#%%
#%% Recall
# manually enter numbers for recall@10 [ssl, r1, r2, r0, r3]
x_recall = ['exp6M', 'expS1', 'expS2', 'expS3' , 'expS0']
r_image = [0.2, 0.7, 5.2 , 9.8 , 5.0 ]
r_speech = [0.2, 0.9, 6.1, 11.7 , 6.4]
results_recall = [r_image, r_speech]

# S1: 6M: Audio R@10 0.009 Image R@10 0.007     FB: Audio R@10 0.011 Image R@10 0.007
# S2: 6M: Audio R@10 0.061 Image R@10 0.052    FB: Audio R@10 0.051 Image R@10 0.044
# S3: 6M: Audio R@10 0.117 Image R@10 0.098     FB: Audio R@10 0.116 Image R@10 0.092
#................
# S0: 6M: Audio R@10 0.064 Image R@10 0.050     FB: Audio R@10 0.055 Image R@10 0.050



# ABX 

models = ['exphh','expS1', 'expS2', 'expS3', 'expS0']

dict_scores = {}
dict_bs = {}
dict_bl = {}
for model in models:
    scores, bs, bl = read_all_layers_abx (model)
    dict_scores[model] = scores
    dict_bs[model] = bs
    dict_bl[model] = bl
x_abx = []
results_abx = []
for key, value in dict_bs.items():
    print(key)
    x_abx.append(key)
    results_abx.append(value)
    
    
#  Lex
models = ['exp15','expS1', 'expS2', 'expS3', 'expS0']
dict_scores, dict_bs, dict_bl, dict_std = get_best_lexical_score(path_lex, models, 'output.txt')
x_FB = []
scores_lex = []
std_lex = []
for key, value in dict_bs.items():
    print(key)
    x_FB.append(key)
    scores_lex.append(value)
    std_lex.append(dict_std [key])
results_lex = (scores_lex, std_lex)


#  SEM
# Measure 3
Snames = ["S1_aL_vO","S2_aL_vO", "S3_aL_vO" , "S0_aL_vO" ]
s_O, std_O, cat_O = find_measure3 (path_sem ,Snames)
# Snames = ["S1_aL_vM","S0_aL_vM","S2_aL_vM","S3_aL_vM"  ]
# s_M, cat_M = find_measure3 (S_path ,Snames)
Snames = ["S1_aL_vB","S2_aL_vB","S3_aL_vB" ,"S0_aL_vB" ]
s_B, std_B, cat_B = find_measure3 (path_sem ,Snames)

# for 6 months ssl
Snames = ["S_aL_vO","S_aL_vM","S_aL_vB"  ]
path_sem = os.path.join(root, "semtest/S/ssl/exp15")
s_ssl, std_ssl, cat_ssl = find_measure3 (path_sem ,Snames)

# for SSL (later compute this)
s_O.insert(0, s_ssl[0])
s_B.insert(0, s_ssl[2])

std_O.insert(0, std_ssl[0])
std_B.insert(0, std_ssl[2])
results_sem = [(s_O, std_O), (s_B, std_B)]
#%%
# plotting
plot_all (names , results_recall, results_abx, results_lex, results_sem)