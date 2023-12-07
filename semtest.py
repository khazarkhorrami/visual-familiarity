


root = "/worktmp2/hxkhkh/current/semtest/"
mtype = 'DINO'# RCNN


Sname_to_hname = {}
Sname_to_hname ["S1_aL_vO"] = "8 months"
Sname_to_hname ["S0_aL_vO"] = "10 months uniform"
Sname_to_hname ["S2_aL_vO"] = "10 months"
Sname_to_hname ["S3_aL_vO"] = "12 months"

#%%
from scipy import stats
from scipy.stats import ranksums
import numpy as np
import os
from matplotlib import pyplot as plt
import json

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
    
    
#%% random S
# S = np.random.randn(1600, 1600)

#%%
# measurement 0: recall@10
def recall10 (S):   
    hits = 0
    for counter in range(len(S)):
        row = S[counter, :]
        row_sorted = list(np.argsort((row))[::-1])
        inspection_window = row_sorted [0:10]
        if counter in inspection_window:
            hits += 1
            
    recall =  round(hits/ 1600 , 3)
    print(f"Recall@10 : {recall}")
    return recall
#%%
# measurement 1
# random 0.25 ( 1/80 *20)

def measure_1 (S):
    scores_cats = []
    for counter in range(len(S)):
        row = S[counter, :]
        green_window_index = dict_word_to_obj_ind [counter]
        #green_window = [row [i] for i in green_window_index]
        
        #red_window_index = [i for i in range(0,1600) if i not in green_window_index]
        #red_window = [row [i] for i in red_window_index]
        
        row_sorted =  list(np.argsort((row))[::-1]) # as bozorg be koochik
        inspection_window = row_sorted[0:20]
        score_row = len(set(inspection_window).intersection(green_window_index))
        scores_cats.append(score_row/20)
        
    m =  round(np.average(scores_cats) ,3)   
    print(f" Measurement 1 : {m} ")
    return m
#%%
# measurement 2

#%%
# measurement 3

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
#%%
def find_measure1 (S_path, Snames):  
    ms = [] 
    for Sname in Snames:
        P = os.path.join(S_path , Sname)
        S = np.load( P + ".npy")  
        m = measure_1 (S)
        ms.append(m)
    return ms

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

def find_all_measures (S_path, Snames):  
    ms = []
    ss = []
    scats = []
    
    for Sname in Snames:
        P = os.path.join(S_path , Sname)
        S = np.load( P + ".npy")  
        m = measure_1 (S)
        s, scat  = measure_3 (S)
        ms.append(m)
        ss.append(s)
        scats.append(scat)
    return ms, ss, scats


#%%
def plotbar_multi (names, results , title, yname, cl):
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

def plotbar_multi_all (names, results , title, yname, cl):
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
            edgecolor ='grey', label =title[0])
    plt.bar(br2, results[1], color ='g', width = barWidth,
            edgecolor ='grey', label =title[1])
    plt.bar(br3, results[2], color ='grey', width = barWidth,
            edgecolor ='grey', label =title[2])
    plt.title("comparison between all pretrained versions" + '\n', fontweight ='bold', fontsize = 28)
    plt.ylabel('semantic test ' + '(' + yname + ')' + '\n', fontweight ='bold',fontsize=24)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 20)
    
    #plt.ylim(0,1) 
    plt.yticks(fontsize=20)
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + 'all' + yname + '.png' ,  format = 'png' )
    plt.show()
    
def plotbar_single (names, results , title, yname , cl):
    barWidth = 0.25
    plt.subplots(figsize =(12, 12))  
    
    n = len(results)
    
    br1 = np.arange(n)
    ychance = np.ones(len(br1))*cl
    plt.plot(br1,ychance, color ='orange', label='chance level', linewidth=1.5)
    
    plt.bar(br1, results, color ='b', width = barWidth,
            edgecolor ='grey')
    plt.title(title + '\n', fontweight ='bold', fontsize = 28)
    plt.ylabel('semantic test ' + '(' + yname + ')' + '\n', fontweight ='bold',fontsize=24)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 20)
    plt.yticks(fontsize=20)
    #plt.ylim(0,1) 
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + title + yname + '.png' ,  format = 'png' )
    plt.show()



###############################################################################
#%% For correlations and T-test
ttype = 'expFB'
title = mtype + ', Pre' + ttype[-2:]
S_path = os.path.join(root, 'S', mtype, ttype)
Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]


# write results on json file (for Okko)
# import json
# data = {}
# data [mtype] = {}
# p = os.path.join(root, 'results', mtype)
# data[mtype]['8 months'] = cat_O[0]
# data[mtype]['uniform'] = cat_O[1]
# data[mtype]['10 months'] = cat_O[2]
# data[mtype]['12 months'] = cat_O[3]
# file_json = p + ".json"
# with open(file_json, "w") as fp:
#     json.dump(data,fp) 
    
# with open(file_json, "r") as fp:
#     d = json.load(fp) 
###############################################################################
#%% T-test
# apply t-test class by class
# T-test will tell you which classes are doing above chance

# compare the distribution of green box with red box for each class
# Snames = ["S_aL_vM","S6M_aL_vM", "S1_aL_vM","S2_aL_vM","S3_aL_vM","S0_aL_vM"  ]
# s_M, std_M, cat_M = find_measure3 (path_sem ,Snames)

# #s_O, cat_O = find_measure3 (S_path ,Snames)
# dict_ttest_all = {}
# for counter, Sname in enumerate(Snames):
#     P = os.path.join(S_path , Sname)
#     S = np.load( P + ".npy")  

#     dict_ttest = find_t_stat (S)
#     dict_ttest_all[Sname] = dict_ttest
#     print(Sname_to_hname [Sname])
#     for key, value in dict_ttest.items():
#         r,p = value
#         if p > 0.05:
#             print("chance level performance for " + key)
#             print(cat_M[counter][key])
#     print("............................................")
#%% Correlations
kh


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

meta_0 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/sub0_meta.json"
meta_1 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/sub1_meta.json"
meta_2 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/sub2_meta.json"
meta_3 = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets/sub3_meta.json"

meta_file = meta_0
d_areas_0, d_freq_0 = return_meta (meta_0)
d_areas_1, d_freq_1 = return_meta (meta_1)
d_areas_2, d_freq_2 = return_meta (meta_2)
d_areas_3, d_freq_3 = return_meta (meta_3)

d_meta_0 = d_freq_0
d_meta_1 = d_freq_1
d_meta_2 = d_freq_2
d_meta_3 = d_freq_3

corr_name = 'corr_freq_pearson.png'
xlab = '\nobject frequencies'
#%%
o8m = []
s8m = []
f8m = []
for item in cat_O[0]:
    o = item[0]
    s = item[1]
    f = d_meta_1[o]
    if o != "person":
        o8m.append(o)
        s8m.append(s)
        f8m.append(f)
res8m = stats.pearsonr(f8m, s8m)
print(res8m)

o10mU = []
s10mU = []
f10mU = []
for item in cat_O[1]:
    o = item[0]
    s = item[1]
    f = d_meta_0[o]
    if o != "person":
        o10mU.append(o)
        s10mU.append(s)
        f10mU.append(f)
res10mU = stats.pearsonr(f10mU, s10mU)
print(res10mU)

o10m = []
s10m = []
f10m = []
for item in cat_O[2]:
    o = item[0]
    s = item[1]
    f = d_meta_2[o]
    if o != "person":
        o10m.append(o)
        s10m.append(s)
        f10m.append(f)
res10m = stats.pearsonr(f10m, s10m)
print(res10m)


o12m = []
s12m = []
f12m = []
for item in cat_O[3]:
    o = item[0]
    s = item[1]
    f = d_meta_3[o]
    if o != "person":
        o12m.append(o)
        s12m.append(s)
        f12m.append(f)
res12m = stats.pearsonr(f12m, s12m)
print(res12m)

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

plt.savefig(os.path.join(root , 'results' , 'correlations', corr_name) ,  format = 'png' )
#%% individual RCNN
# ttype = 'expFB'
# title = mtype + ', Pre' + ttype[-2:]
# S_path = os.path.join(root, 'S', mtype, ttype)
# #if mtype == "vfsubsets":
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# m_O = find_measure1 (S_path ,Snames)

# # Measure 3
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# s_O, cat_O = find_measure3 (S_path ,Snames)

# # plotting
# names = ["8\n months", "10\n months\n (uniform)", "10\n months","12\n months"]

# results = m_O 
# plotbar_single (names, results, title , yname = 'm1', cl= 0.0125)

# results = s_O
# plotbar_single (names, results, title, yname = 'm3', cl = 0.50)

#%% individual DINO
# ttype = 'expFB'
# title = mtype + ', Pre' + ttype[-2:]
# S_path = os.path.join(root, 'S', mtype, ttype)
# #if mtype == "vfsubsets":
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# m_O = find_measure1 (S_path ,Snames)
# Snames = ["S1_aL_vM","S0_aL_vM","S2_aL_vM","S3_aL_vM"  ]
# m_M = find_measure1 (S_path ,Snames)
# Snames = ["S1_aL_vB","S0_aL_vB","S2_aL_vB","S3_aL_vB"  ]
# m_B = find_measure1 (S_path ,Snames)

# # Measure 3
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# s_O, cat_O = find_measure3 (S_path ,Snames)
# Snames = ["S1_aL_vM","S0_aL_vM","S2_aL_vM","S3_aL_vM"  ]
# s_M, cat_M = find_measure3 (S_path ,Snames)
# Snames = ["S1_aL_vB","S0_aL_vB","S2_aL_vB","S3_aL_vB"  ]
# s_B, cat_B = find_measure3 (S_path ,Snames)

# # plotting
# names = ["8\n months", "10\n months\n (uniform)", "10\n months","12\n months"]

# results = [m_O, m_M, m_B ]
# plotbar_multi (names, results, title , yname = 'm1', cl= 0.0125)

# results = [s_O, s_M, s_B ]
# plotbar_multi (names, results, title, yname = 'm3', cl = 0.50)

#%% All 
# title = []
# ttype = 'expFB'
# title.append('Pre' + ttype[-2:])
# S_path = os.path.join(root, 'S', mtype, ttype)
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# m_FB = find_measure1 (S_path ,Snames)
# s_FB, cat_FB = find_measure3 (S_path ,Snames)

# ttype = 'exp6M'
# title.append('Pre' + ttype[-2:])
# S_path = os.path.join(root, 'S', mtype, ttype)
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# m_6M = find_measure1 (S_path ,Snames)
# s_6M, cat_6M = find_measure3 (S_path ,Snames)

# ttype = 'expR'
# title.append('Pre' + ttype[-1:])
# S_path = os.path.join(root, 'S', mtype, ttype)
# Snames = ["S1_aL_vO","S0_aL_vO","S2_aL_vO","S3_aL_vO"  ]
# m_R = find_measure1 (S_path ,Snames)
# s_R, cat_R = find_measure3 (S_path ,Snames)

# # plotting
# names = ["8\n months", "10\n months\n (uniform)", "10\n months","12\n months"]

# results = [m_FB, m_6M, m_R ]
# plotbar_multi_all (names, results, title , yname = 'm1', cl= 0.0125)

# results = [s_FB, s_6M, s_R ]
# plotbar_multi_all (names, results, title, yname = 'm3', cl = 0.50)

