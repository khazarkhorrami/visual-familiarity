import numpy as np
import os
import json
root = "/worktmp2/hxkhkh/current/"

########################################################################### 
def define_cats (root):
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
    return categories_all

def define_indexes():   
    categories_all = define_cats (root)
    tt = []
    categories = []
    for t in range(0,1600,20):
        x = list(range(t,t+20))
        tt.append(x)
        categories.append(categories_all[t])
    return tt, categories

def find_word_inds():
    tt, categories = define_indexes()
    dict_word_to_obj_ind = {}    
    for chunk in tt:
        for element in chunk:
            dict_word_to_obj_ind [element] = chunk   
    return dict_word_to_obj_ind

         
def find_degree_per_row (row_index, S):
    dict_word_to_obj_ind = find_word_inds ()
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
        z_sorted = list(np.argsort(z)) # ascending
        argq = z_sorted.index(0) #978
        d = argq / 1599 
        degree_row.append(round(d,3))
    return np.average(degree_row)

def find_degree_per_category (category_index, S):
    tt, categories = define_indexes()
    chunk_rows = tt [category_index]
    d_category = []
    for row_index in chunk_rows:
        d = find_degree_per_row (row_index, S)
        d_category.append(round(d,3))
    return d_category

def calculate_semtest (S): 
    tt, categories = define_indexes()
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
    
    scores_cats_sorted = sorted(scores_cats_average.items(), key=lambda x:x[1], reverse=True)
    print(f"Semtest : {score_mean}")
    print(scores_cats_sorted[0:5])    
    return score_mean, score_std, scores_cats_average, scores_cats_sorted
        

S_path = os.path.join(root, "semtest", "S", "exampleS.npy")
S = np.load(S_path)   
score_mean, score_std, scores_cats_average, scores_cats_sorted = calculate_semtest (S)