import numpy as np

S_path = '../../semtest/Smatrix/'

S = np.load(S_path + "exampleS.npy")

#%%

dict_word_to_obj_ind = {}

tt = []
for t in range(0,1600,20):
    x = list(range(t,t+20))
    tt.append(x)
    
for chunk in tt:
    for element in chunk:
        dict_word_to_obj_ind [element] = chunk
    
    

#%%
# measurement 1

#%%
# measurement 2

#%%
# measurement 3