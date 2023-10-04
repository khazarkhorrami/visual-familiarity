


root = "/worktmp2/hxkhkh/current/semtest/"
#%%
import numpy as np
import os
from matplotlib import pyplot as plt
import json
S_path = os.path.join(root, 'Smatrix')
  
    
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
    
#%%
def find_measure1 (Snames):  
    ms = [] 
    for Sname in Snames:
        S = np.load(S_path + Sname + ".npy")  
        m = measure_1 (S)
        ms.append(m)
    return ms

def find_measure3 (Snames):  
    ss = []
    scats = []   
    for Sname in Snames:
        S = np.load(S_path + Sname + ".npy")          
        s, scat  = measure_3 (S)
        ss.append(s)
        scats.append(scat)
    return ss, scats

def find_all_measures (Snames):  
    ms = []
    ss = []
    scats = []
    
    for Sname in Snames:
        S = np.load(S_path + Sname + ".npy")  
        m = measure_1 (S)
        s, scat  = measure_3 (S)
        ms.append(m)
        ss.append(s)
        scats.append(scat)
    return ms, ss, scats



#%%
def plotbar_multi (names, results , title):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results[0])
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
     
    
    plt.bar(br1, results[0], color ='r', width = barWidth,
            edgecolor ='grey', label ='Original image')
    plt.bar(br2, results[1], color ='g', width = barWidth,
            edgecolor ='grey', label ='Masked image')
    plt.bar(br3, results[2], color ='b', width = barWidth,
            edgecolor ='grey', label ='Blurred image')
     

    plt.ylabel(title + '\n', fontweight ='bold', fontsize = 20)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 20)
     
    plt.legend(fontsize = 18)

    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + title + '.png' ,  format = 'png' )
    plt.show()

#%%


# Measure 1
Snames = ["S1_aL_vO", "S2_aL_vO","S3_aL_vO"  ]
m_O = find_measure1 (Snames)
Snames = ["S1_aL_vM", "S2_aL_vM","S3_aL_vM"  ]
m_M = find_measure1 (Snames)
Snames = ["S1_aL_vB", "S2_aL_vB","S3_aL_vB"  ]
m_B = find_measure1 (Snames)

# Measure 3
Snames = ["S1_aL_vO", "S2_aL_vO","S3_aL_vO"  ]
s_O, cat_O = find_measure3 (Snames)
Snames = ["S1_aL_vM", "S2_aL_vM","S3_aL_vM"  ]
s_M, cat_M = find_measure3 (Snames)
Snames = ["S1_aL_vB", "S2_aL_vB","S3_aL_vB"  ]
s_B, cat_B = find_measure3 (Snames)

#%%
# plotting
names = ["subset 1\n(2 months)", "subset 2\n(4 months)","subset 3\n(6 months)"]

results = [m_O, m_M, m_B ]
plotbar_multi (names, results, "measurement_1")

results = [s_O, s_M, s_B ]
plotbar_multi (names, results, "measurement_3")