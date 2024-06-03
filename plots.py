
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import csv

# global variables
root = "/worktmp2/hxkhkh/projects/"
path_abx = os.path.join(root, 'ZeroSpeech/output/AC/DINO/exp6M/')
path_lex = os.path.join(root, "lextest/output/CDI/DINO/exp6M")
path_sem = os.path.join(root, "semtest/S/DINO/exp6M")
path_save = "/worktmp2/hxkhkh/projects/FaST/papers/vf/"
dtype = 'CDI'
mtype = 'DINO'



names = ["baseline", "auditory-only ", " ","8\nmo", "10\nmo", "12\nmo", "",  "10\nmo \n(uniform)"]



# functions
def plot_all_phone_word_wmeaning (names , results_recall, results_abx, results_lex, results_sem):
    recall_cl = 0.2
    abx_cl = 50
    abx_mfcc = 10.95
    lex_cl = 1/89
    lex_logmel = 0.17
    sem_cl = 0.5
    plt.figure(figsize =(32,12))
    f_leg = 28
    f_ticks = 26
    f_ylabel = 34
    chancecolor = "brown"
    c1 = ["royalblue"]
    c2 = ["seagreen"]  
    c3 = ["palevioletred"]
    #c3 = ["peru"]
    barWidth = 0.35 
    
    
    ######### ABX
    
    plt.subplot(1,3,1)
    n = len(results_abx)
    br1 = np.arange(n)
    #ylogmel = np.ones(len(br1))*abx_mfcc
    #plt.plot(br1,ylogmel, color ='blue', label='MFCC features', linewidth=5)
    ychance = np.ones(len(br1))*abx_cl
    plt.plot(br1,ychance, color = chancecolor, label='Chance level' , linewidth=5)
    bars = plt.bar(br1, results_abx, color = c1, width = barWidth, edgecolor='black', ecolor='black', capsize=10, linewidth = 4, alpha = 0.8) 
    plt.ylabel('Phonemic discrimination error rate (%)',  fontsize = f_ylabel -2) #fontweight ='bold',
    plt.xlabel('\n Age (months)', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,51)    
    plt.grid()    
    bars[1].set_hatch('//')
    bars[0].set_hatch('.')    
    bars[5].set_alpha(0.4)
    
    # legend
    barWidth = 0.01
    plt.bar(br1, br1*0, edgecolor='black', facecolor='none', hatch='.',linewidth = 4 , width = barWidth, label='Baseline', alpha=0.8)
    plt.bar(br1, br1*0, edgecolor='black', facecolor='none', hatch='//',linewidth = 4 , width = barWidth, label='Auditory learning', alpha=0.8)
    plt.bar(br1, br1*0, edgecolor='black', facecolor='gray', hatch='',linewidth = 4 , width = barWidth, label='Auditory + audiovisual learning', alpha=0.5) 
    plt.legend(fontsize = f_leg, loc="upper right",  bbox_to_anchor=(2.0,1.35),  framealpha=0.1)
    barWidth = 0.35
    
    ######### lEX
    (scores_lex, std_lex) =  results_lex
    plt.subplot(1,3,2)
    n = len(scores_lex)  
    br1 = np.arange(n)
    ychance = np.ones(len(br1))*lex_cl
    #ylogmel = np.ones(len(br1))*lex_logmel
    plt.plot(br1, ychance, color = chancecolor, linewidth=5)
    #plt.plot(br1,ylogmel, color ='blue', label='log-Mel features', linewidth=5)
    bars = plt.bar(br1, scores_lex,  yerr=std_lex , color = c2, width = barWidth,
            edgecolor ='black', align='center', ecolor='black', capsize=10, linewidth = 4, alpha = 0.8)
    
    plt.ylabel('Lexical discrimination score (0-1)' ,  fontsize = f_ylabel-2) #fontweight ='bold',
    plt.xlabel('\n Age (months)\n', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1.1) 
    plt.grid()
    bars[1].set_hatch('//')
    #bars.set_linewidth(4)
    bars[0].set_hatch('.')
    #bars.set_linewidth(4)  
    bars[5].set_alpha(0.4)
    
    ######### sem
    [scores_sem, std_sem] = results_sem 
    plt.subplot(1,3,3)  
    
    n = len(scores_sem)  
    br1 = np.arange(n)
    ychance = np.ones(len(br1))*sem_cl
    plt.plot(br1,ychance, color = chancecolor, linewidth=5)
    bars = plt.bar(br1, scores_sem,  yerr = std_sem , color = c3, width = barWidth,
            edgecolor ='black', align='center', ecolor='black', capsize=10, linewidth = 4, alpha = 0.8)
    
    plt.ylabel('Word meaning score (0-1)' , fontsize = f_ylabel-2) #fontweight ='bold',
    plt.xlabel('\n Age (months)\n', fontsize = f_ylabel)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = f_ticks)
    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,1.1) 
    #plt.legend(fontsize = f_leg, framealpha=0.1)#, framealpha=0.1)
    plt.grid()
    bars[1].set_hatch('//')
    #bars.set_linewidth(4)
    bars[0].set_hatch('.')
    #bars.set_linewidth(4)
    
    bars[5].set_alpha(0.4)
    #plt.tight_layout(pad=6.0)
    #plt.savefig(path_save +  'results_nature.pdf' ,  format = 'pdf', bbox_inches='tight' ) #, bbox_inches='tight'
    

# functions
def plot_all (names , results_recall, results_abx, results_lex, results_sem):
    
    plt.figure(figsize =(32,12))
    
    f_ticks = 28
    f_ylabel = 34 
    c2 = ["seagreen"]  
    barWidth = 0.55
    
    ######### ABX

    ######### lEX
    (scores_lex, std_lex) =  results_lex
    plt.subplot(1,2,1)
    n = len(scores_lex)  
    br1 = list(np.arange(n+2))
    scores_lex_m = scores_lex[0:2]
    scores_lex_m.extend([0])
    scores_lex_m.extend(scores_lex[2:-1])
    scores_lex_m.extend([0])
    scores_lex_m.extend([scores_lex[-1]])
    scores_lex_m = np.array (scores_lex_m)
    scores_lex_m = scores_lex_m * 100
    bars = plt.bar(br1, scores_lex_m,  color = c2, width = barWidth,
            edgecolor ='black', align='center', ecolor='black', capsize=10, linewidth = 4, alpha = 1)
    
    plt.ylabel('Word discrimination score (%)' ,  fontsize = f_ylabel-2, fontweight ='bold') #fontweight ='bold',
    #plt.xlabel('\n Age (months)\n', fontsize = f_ylabel)
    plt.xticks([r for r in range(n+2)], names, fontweight ='bold',fontsize = f_ticks)
    #plt.xticks(rotation=45)
    plt.gca().xaxis.get_major_ticks()[0].label.set_rotation(45)  # Rotate only the first
    plt.gca().xaxis.get_major_ticks()[1].label.set_rotation(45)  # Rotate only the first

    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,101) 
    ax = plt.gca()
    ax.yaxis.grid()
    # green shades: #(0.311, 0.426, 0.263, 0.5)
    # (0.592, 0.776, 0.369, 0.5)
    # # 0.3141, 0.4188, 0.2671, 0.5)
    bars[0].set_color ((0.6, 0.6, 0.6, 0.5))
    bars[1].set_color((0.95, 0.76, 0.20, 0.5))
    bars[3].set_color((0.6, 0.7, 0.4, 1))  
    bars[4].set_color((0.6, 0.7, 0.4, 1))
    bars[5].set_color((0.6, 0.7, 0.4, 1)) 
    bars[7].set_color((.612,.804,.80))#(0.725, 0.835, 0.678, 0.5))
    bars[2].set_alpha(0.01)
    bars[6].set_alpha(0.01)
    #bars[7].set_alpha(a)
    
    ######### sem
    [scores_sem, std_sem] = results_sem 
    plt.subplot(1,2,2)  
    
    n = len(scores_sem)  
    br2 = list(np.arange(n+2))
    scores_sem_m = scores_sem[0:2]
    scores_sem_m.extend([0])
    scores_sem_m.extend(scores_sem[2:-1])
    scores_sem_m.extend([0])
    scores_sem_m.extend([scores_sem[-1]])
    scores_sem_m = np.array (scores_sem_m)
    scores_sem_m = scores_sem_m * 100
    bars = plt.bar(br2, scores_sem_m,  color = c2, width = barWidth,
            edgecolor ='black', align='center', ecolor='black', capsize=10, linewidth = 4, alpha = 1)
    
    plt.ylabel('Word meaning score (%)' , fontsize = f_ylabel-2, fontweight ='bold') #fontweight ='bold',
    #plt.xlabel('\n Age (months)\n', fontsize = f_ylabel)
    plt.xticks([r for r in range(n+2)], names, fontweight ='bold',fontsize = f_ticks)
    plt.gca().xaxis.get_major_ticks()[0].label.set_rotation(45)  # Rotate only the first
    plt.gca().xaxis.get_major_ticks()[1].label.set_rotation(45)  # Rotate only the first

    plt.yticks(fontsize = f_ticks)
    plt.ylim(0,101) 
    ax = plt.gca()
    ax.yaxis.grid()
    bars[0].set_color ((0.6, 0.6, 0.6, 0.5))
    bars[1].set_color((0.95, 0.76, 0.20, 0.5))
    # bars[3].set_color((0.592, 0.776, 0.369, 0.5))
    # bars[4].set_color((0.592, 0.776, 0.369, 0.5))
    # bars[5].set_color((0.592, 0.776, 0.369, 0.5)) 
    bars[3].set_color((0.6, 0.7, 0.4, 1))  
    bars[4].set_color((0.6, 0.7, 0.4, 1))
    bars[5].set_color((0.6, 0.7, 0.4, 1))
    bars[7].set_color((.612,.804,.80))
    bars[2].set_alpha(0.01)
    bars[6].set_alpha(0.01)

    #plt.tight_layout(pad=6.0)
    plt.savefig(path_save +  'results_color.pdf' ,  format = 'pdf', bbox_inches='tight' ) #, bbox_inches='tight'

# all
ADS = np.array([8.06, 4.38, 10.83, 11.08, 19.87] )
IDS = np.array([7.43, 7.43  ,7.43,7.43, 7.43] )
IDS = np.array([5.05, 7.43  ,7.43,7.43, 15.09] )

# 0,3,6,9,12 months
# awakes = [24-16, 24-15,24-14, 24-14,24-14]
awakes = [8, 9, 10, 10, 10]
awakes_12m = np.mean(awakes) # 9.4
awakes_6m = np.mean(awakes[0:3]) # 9
awakes_9m = np.mean(awakes[0:4]) # 9.25
# sleep hours for newborns: 16 hours ( awake: 8 hours)
# sleep hours for 6 months old:. 14 hours (awake: 10 hours)

# UK
ADS = np.array([8.06])
IDS = np.array([5.05])
S_per_hour_average = np.mean(ADS + IDS)
S_pre = (S_per_hour_average) * (11/60) * 15 * 30 # 1037 ( 15 months NA, 11 hours awake)
S_pre 

# NA
ADS = np.array([8.06])
IDS = np.array([5.05])
S_per_hour_average = np.mean(ADS + IDS)
S_pre = (S_per_hour_average) * (11/60) * 15 * 30 # 1037 ( 15 months NA, 11 hours awake)
S_pre 

#Yélî Dnye
ADS = np.array([19.87])
IDS = np.array([15.09])
S_per_hour_average = np.mean(ADS + IDS)
S_pre = (S_per_hour_average) * (10/60) * 6 * 30 # 1037 ( 6 months Yélî Dnye, 10 hours awake)
S_pre 
#%%
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
    layer_names = ['L1','L2','L3','L4','L5','L6','L7' ]
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
    layer_names = ['L0','L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11' ]
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
    scores_cats_average = {}
    for category_index in range(80):
        d_cat = find_degree_per_category (category_index, S)         
        scores_degree_all.extend(d_cat)
        scores_degree_cats.append(d_cat)
        scores_cats_average[categories[category_index]] = round (np.average(d_cat),3)
    
    score_mean = round(np.average(scores_degree_all) , 3)
    score_std = round(np.std(scores_degree_all) , 3)
    
    score_sorted = sorted(scores_cats_average.items(), key=lambda x:x[1], reverse=True)
    print(f"Measurement 3 : {score_mean}")
    print(score_sorted[0:5])    
    return score_mean, score_std, scores_cats_average , score_sorted
        
def find_measure3 (S_path, Snames):  
    ss = []
    ss_std = []
    cats = []
    scats = []
    for Sname in Snames:
        P = os.path.join(S_path , Sname)
        S = np.load( P + ".npy")          
        s, s_std, cat, scat = measure_3 (S)
        ss.append(s)
        ss_std.append(s_std)
        cats.append(cat)
        scats.append(scat)
    return ss,ss_std, cats, scats  

# Snames = ["S_aL_vO"]
# path_sem = os.path.join(root, "semtest/S/ssl/exp15")
# S_path = path_sem

# S = np.random.randn(1600, 1600)


#%% Recall
# manually enter numbers for recall@10 [ssl, r1, r2, r0, r3]
x_recall = ['baseline','exp6M', 'expS1', 'expS2', 'expS3' , 'expS0']
r_image = [0.2, 0.2, 0.7, 5.2 , 9.8 , 5.0 ]
r_speech = [0.2, 0.2, 0.9, 6.1, 11.7 , 6.4]
results_recall = [r_image, r_speech]

# S1: 6M: Audio R@10 0.009 Image R@10 0.007     FB: Audio R@10 0.011 Image R@10 0.007
# S2: 6M: Audio R@10 0.061 Image R@10 0.052    FB: Audio R@10 0.051 Image R@10 0.044
# S3: 6M: Audio R@10 0.117 Image R@10 0.098     FB: Audio R@10 0.116 Image R@10 0.092
#................
# S0: 6M: Audio R@10 0.064 Image R@10 0.050     FB: Audio R@10 0.055 Image R@10 0.050



# ABX 

models = ['baseline', 'exp15','expS1', 'expS2', 'expS3', 'expS0']

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
models = ['baseline', 'exp15','expS1', 'expS2', 'expS3', 'expS0']
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
# Snames = ["S_aL_vO", "S6M_aL_vO", "S1_aL_vO","S2_aL_vO", "S3_aL_vO" , "S0_aL_vO" ]
# s_O, std_O, cat_O, scat_O  = find_measure3 (path_sem ,Snames)
Snames = ["S_aL_vM","S6M_aL_vM", "S1_aL_vM","S2_aL_vM","S3_aL_vM","S0_aL_vM"  ]
s_M, std_M, cat_M, scat_M = find_measure3 (path_sem ,Snames)
# Snames = ["S_aL_vB","S6M_aL_vB", "S1_aL_vB","S2_aL_vB","S3_aL_vB" ,"S0_aL_vB" ]
# s_B, std_B, cat_B, scat_B = find_measure3 (path_sem ,Snames)

results_sem = [s_M, std_M]
kh
#%%
# plotting
plot_all (names , results_recall, results_abx, results_lex, results_sem)

#%% plot catgories: cat_M[4] or for sorted one: scat_M
(0.6, 0.7, 0.4, 1)
colors = [(0.6, 0.6, 0.6), (0.95, 0.76, 0.20), (0.592, 0.776, 0.369),
          (0.592, 0.776, 0.369), (0.592, 0.776, 0.369), (.612,.804,.80)]

colors = [(0.6, 0.6, 0.6), (0.95, 0.76, 0.20), (0.6, 0.7, 0.4, 1),
          (0.6, 0.7, 0.4, 1), (0.6, 0.7, 0.4, 1), (.612,.804,.80)]

models = ['baseline\n', 'auditory-only\n','8 mo\n', '10 mo\n', '12 mo\n', '10 mo (uniform)\n']
ychance = np.ones(80)*50
plt.figure(figsize =(45,8))
for i in range(len(scat_M)):
    scores = []
    names = []
    for counter, value in enumerate(scat_M[i]):
        scores.append(value[1]*100)
        names.append(value[0])
    plt.subplot(1,6,i+1)
    plt.ylim(0, 101)    
    plt.plot(scores, color = colors[i], label=models[i], linewidth=12)
    #test = plt.hist(scores, bins= 10, label=models[i])
    plt.xticks(fontsize= 28)
    plt.yticks(fontsize= 28)
    plt.plot(ychance, color = 'gray', linewidth=3)
    #plt.legend(fontsize= 36)
    plt.title(models[i],fontsize= 40, fontweight ='bold')
    plt.grid()
    
    # ax = plt.gca()
    # ax.set_xticklabels([])
    # ax.grid(True)
#bars[1].set_color((0.95, 0.76, 0.20, 0.5))

# plt.suptitle('Distribution of word meaning scores (%) over all categories\n\n', fontsize= 48, fontweight ='bold')
plt.savefig(path_save +  'cats_curves_color.pdf' ,  format = 'pdf', bbox_inches='tight') #, bbox_inches='tight'
#plt.savefig(path_save +  'cats_dists.pdf' ,  format = 'pdf', bbox_inches='tight') #, bbox_inches='tight'

#%%
plt.figure(figsize =(24,6))
br = np.arange(80)
scores_12m = []
names = []
for counter, value in enumerate(scat_M[4]):
    scores_12m.append(value[1]*100)
    names.append(value[0])

plt.bar(br, scores_12m, color=(0.5, 0.26, 0.31, 0.5) , alpha= .4)#"lightblue" # color=(0.592, 0.776, 0.369)
plt.xticks([r for r in range(80)], names, fontsize = 16, fontweight ='bold')
plt.xticks(rotation=90)
plt.yticks(fontsize= 18)
plt.ylabel('Word meaning score (%)\n\n(12 months)', fontsize= 24, fontweight ='bold')
ax = plt.gca()
ax.yaxis.grid()
ax.margins(x=.01)
plt.savefig(path_save +  'cats_bar_2.pdf' ,  format = 'pdf', bbox_inches='tight' ) #, bbox_inches='tight'

#%%
from scipy.io import savemat
mydict = {"ages": names, "recall": results_recall, "ABX": results_abx ,"Lextest": results_lex,"Semtest": results_sem}
savemat(path_save + "Results_6MPre.mat", mydict)

data = {}
mtype = 'DINO'
data [mtype] = {}
data[mtype]['0 months'] = cat_M[0]
data[mtype]['6 months'] = cat_M[1]
data[mtype]['8 months'] = cat_M[2]
data[mtype]['10 months'] = cat_M[3]
data[mtype]['12 months'] = cat_M[4]
data[mtype]['uniform'] = cat_M[5]


p = os.path.join(path_save, "Semtest_categories" )
file_json = p  +  ".json"
with open(file_json, "w") as fp:
    json.dump(data,fp) 

# testing
with open(file_json, "r") as fp:
    d = json.load(fp) 

#%%
# for Okko
# write category-based results on json file (for Okko)
data = {}
mtype = 'DINO'
data [mtype] = {}
data[mtype]['0 months'] = scat_M[0]
data[mtype]['6 months'] = scat_M[1]
data[mtype]['8 months'] = scat_M[2]
data[mtype]['10 months'] = scat_M[3]
data[mtype]['12 months'] = scat_M[4]
data[mtype]['uniform'] = scat_M[5]


p = os.path.join(path_save, 'forOkko', "results_sem_categories" )
file_json = p  +  ".json"
with open(file_json, "w") as fp:
    json.dump(data,fp) 

# testing
with open(file_json, "r") as fp:
    d = json.load(fp) 
#%% t-test

from scipy.stats import ranksums
def find_t_stat (S):
    
    dict_ttest = {}
    for category_index in range(80):
        
        ind_green = tt [category_index]
        S_green = S[ind_green, :][:,ind_green]
        
        ind_red =  [ind for ind in range(1600) if ind not in ind_green ]
        S_red = S[ind_green, :][:,ind_red]
        
        D_green = S_green.flatten()
        D_red = S_red.flatten()
        (r, p ) = ranksums(D_green, D_red, alternative="greater")
        dict_ttest[categories[category_index]] = (r, p )
    return dict_ttest


# compare the distribution of green box with red box for each class
Snames = ["S_aL_vM","S6M_aL_vM", "S1_aL_vM","S2_aL_vM","S3_aL_vM","S0_aL_vM"  ]
s_M, std_M, cat_M, scat_M = find_measure3 (path_sem ,Snames)

Sname_to_hname = {}
Sname_to_hname ["S_aL_vM"] = "0 months"
Sname_to_hname ["S6M_aL_vM"] = "6 months"
Sname_to_hname ["S1_aL_vM"] = "8 months"
Sname_to_hname ["S2_aL_vM"] = "10 months"
Sname_to_hname ["S3_aL_vM"] = "12 months"
Sname_to_hname ["S0_aL_vM"] = "uniform"


dict_ttest_all = {}
for counter, Sname in enumerate(Snames):
    P = os.path.join(path_sem , Sname)
    S = np.load( P + ".npy")  

    dict_ttest = find_t_stat(S)
    dict_ttest_all[Sname_to_hname[Sname]] = dict_ttest
    print(Sname_to_hname [Sname])
    n_chance = 0
    for key, value in dict_ttest.items():
        r,p = value
        if p > 0.05:
            # print("chance level performance for " + key)
            # print(cat_M[counter][key])
            n_chance +=1
    print ("There were in total " + str(n_chance) + " categories with chance level perormance. ")
    print("............................................")


p = os.path.join(path_save, 'forOkko', "results_ttest" )

file_json = p  +  ".json"
with open(file_json, "w") as fp:
    json.dump(dict_ttest_all,fp) 

# testing
with open(file_json, "r") as fp:
    d = json.load(fp) 
#%%
path_save = "/worktmp2/hxkhkh/current/semtest/results/correlations/"
from scipy import stats

Snames = ["S1_aL_vM","S2_aL_vM","S3_aL_vM","S0_aL_vM"  ]
# Snames = ["S1_aL_vO","S2_aL_vO", "S3_aL_vO" , "S0_aL_vO" ]
# Snames = ["S1_aL_vB","S2_aL_vB", "S3_aL_vB" , "S0_aL_vB" ]
s_M, std_M, cat_M, scat_M = find_measure3 (path_sem ,Snames)

# if you need to check freq correlations considering general data frequencies 
fp_freq = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/dict_words_selected_counts.json"
with open(fp_freq, "r") as fp:
    d_freq = json.load(fp)

# if you need to check correlations considering subset frequencies 
def return_meta (meta_file):
    with open(meta_file, "r") as fp:
        d_meta = json.load(fp)
    d_areas = d_meta['object_areas']
    #d_cap = d_meta['object_cap']
    d_freq = d_meta['object_freq']
    return d_areas, d_freq


meta_1 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub1_meta.json"
meta_2 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub2_meta.json"
meta_3 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub3_meta.json"
meta_0 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/nsub0_meta.json"

# meta_file = meta_0

d_areas_1, d_freq_1 = return_meta (meta_1)
d_areas_2, d_freq_2 = return_meta (meta_2)
d_areas_3, d_freq_3 = return_meta (meta_3)
d_areas_0, d_freq_0 = return_meta (meta_0)


d_meta_1 = d_areas_1
d_meta_2 = d_areas_2
d_meta_3 = d_areas_3
d_meta_0 = d_areas_0

# d_meta_1 = d_freq_1
# d_meta_2 = d_freq_2
# d_meta_3 = d_freq_3
# d_meta_0 = d_freq_0


#corr_name = 'corr_freq_pearson.png'
xlab = '\nobject area'

print('############# for 8 months ###################')
o8m = []
s8m = []
f8m = []
for item in scat_M[0]:
    o = item[0]
    s = item[1]
    f = d_meta_1[o]
    #if o != "person":
    o8m.append(o)
    s8m.append(s)
    f8m.append(f)
res8m = stats.spearmanr(f8m, s8m)
print(res8m)


print('############# for 10 months ###################')
o10m = []
s10m = []
f10m = []
for item in scat_M[1]:
    o = item[0]
    s = item[1]
    f = d_meta_2[o]
    #if o != "person":
    o10m.append(o)
    s10m.append(s)
    f10m.append(f)
res10m = stats.spearmanr(f10m, s10m)
print(res10m)

print('############# for 12 months ###################')
o12m = []
s12m = []
f12m = []
for item in scat_M[2]:
    o = item[0]
    s = item[1]
    f = d_meta_3[o]
    #if o != "person":
    o12m.append(o)
    s12m.append(s)
    f12m.append(f)
res12m = stats.spearmanr(f12m, s12m)
print(res12m)


print('############# for uniform ###################')
o10mU = []
s10mU = []
f10mU = []
for item in scat_M[3]:
    o = item[0]
    s = item[1]
    f = d_meta_0[o]
    #if o != "person":
    o10mU.append(o)
    s10mU.append(s)
    f10mU.append(f)
res10mU = stats.spearmanr(f10mU, s10mU)
print(res10mU)

plt.figure(figsize =(12, 12))
plt.subplot(2,2,1)
plt.scatter(f8m, s8m, label = " (" + str (round(res8m[0],3)) + ' , ' + str(round(res8m[1],3)) + ')' )
plt.title(' 8 months', fontsize = 18)
plt.ylabel('semtest',fontsize = 18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize = 18)
plt.subplot(2,2,2)
plt.scatter(f10mU, s10mU, label = " (" + str (round(res10mU[0],3)) + ' , ' + str(round(res10mU[1],3)) + ')')
plt.title(' 10 months- uniform',fontsize = 18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize = 18)
plt.subplot(2,2,3)
plt.scatter(f10m, s10m, label = " (" + str (round(res10m[0],3)) + ' , ' + str(round(res10m[1],3)) + ')')
plt.title(' 10 months',fontsize = 18)
plt.ylabel('semtest',fontsize = 18)
plt.xlabel(xlab,fontsize = 18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize = 18)
plt.subplot(2,2,4)
plt.scatter(f12m, s12m, label = " (" + str (round(res12m[0],3)) + ' , ' + str(round(res12m[1],3)) + ')')
plt.title(' 12 months',fontsize = 18)
plt.xlabel(xlab,fontsize = 18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize = 18)
plt.savefig(path_save +  'corr_spearman_area_M.pdf' ,  format = 'pdf', bbox_inches='tight' )