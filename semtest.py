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
    
    inspection_window = list(np.argsort(abs(row)) [0:20])
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