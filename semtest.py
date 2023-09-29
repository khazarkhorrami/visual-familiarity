
Sname = "S1b_aO_vO"
Sname = "S2b_aL_vO"
# Sname = "S1b_aO_vm"
# Sname = "S1b_aL_vm"
# Sname = "S1b_aO_vB"
# Sname = "S1b_aL_vB"
#%%
import numpy as np
import json
S_path = '/worktmp2/hxkhkh/current/semtest/Smatrix/'
S = np.load(S_path + Sname + ".npy")  
    
file = "/worktmp2/hxkhkh/current/semtest/semtest_files_pairings.json"
with open(file, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
words = []
objects = []
categories_all = []
for key, value in data.items():
    words.append(key)
    objects.append(value)
    categories_all.append(value.split('_')[0])


#%%

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
S = np.random.randn(1600, 1600)

#%%
# measurement 0: recall@10
hits = 0
for counter in range(len(S)):
    row = S[counter, :]
    row_sorted = list(np.argsort((row))[::-1])
    inspection_window = row_sorted [0:10]
    if counter in inspection_window:
        hits += 1
        
recall =  hits/ 1600  
print(round(recall,3))    
#%%
# measurement 1
# random 0.25 ( 1/80 *20)

scores_cats = []
for counter in range(len(S)):
    row = S[counter, :]
    green_window_index = dict_word_to_obj_ind [counter]
    green_window = [row [i] for i in green_window_index]
    
    red_window_index = [i for i in range(0,1600) if i not in green_window_index]
    red_window = [row [i] for i in red_window_index]
    
    row_sorted =  list(np.argsort((row))[::-1]) # as bozorg be koochik
    inspection_window = row_sorted[0:20]
    score_row = len(set(inspection_window).intersection(green_window_index))
    scores_cats.append(score_row)
    
    
print(round(np.average(scores_cats) ,3 ))
# S3_testm : 0.28125
# S3_testb : 0.2543

# S3b_testb : 0.245
# S3b_testm : 0.253
#%%
# measurement 2

#%%
# measurement 3

def find_degree_per_category (category_index):
    chunk_rows = tt [category_index]
    d_category = []
    for row_index in chunk_rows:
        d = find_degree_per_row (row_index)
        d_category.append(d)
    return d_category

#### here calculates degree for each row 
def find_degree_per_row (row_index):
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
        degree_row.append(d)
    return np.average(degree_row)
##############################################
# example 
# row_index = 50
# d = find_degree_per_row (row_index)

# category_index = 0
# d_cat = find_degree_per_category (category_index)
#%%
   
scores_degree_all = []
scores_degree_cats = []
scores_degree_cats_average = {}

for category_index in range(80):
    d_cat = find_degree_per_category (category_index)
    scores_degree_all.extend(d_cat)
    scores_degree_cats.append(d_cat)
    scores_degree_cats_average[categories[category_index]] = np.average(d_cat)
    
    
print(round(np.average(scores_degree_all) , 3))
    
print(sorted(scores_degree_cats_average.items(), key=lambda x:x[1], reverse=True)[0:5])    
    
    
    