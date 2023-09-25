import numpy as np

S_path = '../../semtest/Smatrix/'

S = np.load(S_path + "Sbest.npy")

#%%

dict_word_to_obj_ind = {}

tt = []
for t in range(0,1600,20):
    x = list(range(t,t+20))
    tt.append(x)
    
for chunk in tt:
    for element in chunk:
        dict_word_to_obj_ind [element] = chunk
    
    
#%% random S
#S = np.random.randint(-1, 1, size=(1600, 1600))
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
    
    inspection_window = list(np.argsort((row)) [0:20])
    score_row = len(set(inspection_window).intersection(green_window_index))
    scores_cats.append(score_row)
    
    
print(np.average(scores_cats))
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
        z_sorted = list(np.argsort(z))
        argq = z_sorted.index(0) #978
        # shifting indexs by 1 to avoid zero in denominator
        after = 1581 -  ( argq + 2)
        degree = after / (argq +1 )
        degree_row.append(degree)
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
scores_degree_cats_average = []

for category_index in range(80):
    d_cat = find_degree_per_category (category_index)
    scores_degree_all.extend(d_cat)
    scores_degree_cats.append(d_cat)
    scores_degree_cats_average.append(np.average(d_cat))
    
    

    
    
    
    
    
    