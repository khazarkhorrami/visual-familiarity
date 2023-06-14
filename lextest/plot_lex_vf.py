import os
import matplotlib.pyplot as plt
import numpy as np

path_input = "/worktmp2/hxkhkh/current/lextest/output/vfsub/"
path_save = '/worktmp2/hxkhkh/current/FaST/plots/lex/'
import csv
#path = '/worktmp2/hxkhkh/current/lextest/output/cls/model7base1T/E5L1/output.txt'

        
def read_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

##################################################################
                        ### vfsub3  ###
################################################################## 
scores_m7base1 = []
model_name = 'vfsub3'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,70]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)


m7base1 = (np.reshape(scores_m7base1, (6,8))).T


##################################################################
                        ### vfsub3fb  ###
##################################################################
scores_m7base2 = []
model_name = 'vfsub3fb'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,10,20,30,40,50,60,70]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base2.append(s)


m7base2 = (np.reshape(scores_m7base2, (12,8))).T

################################################################ Plotting

title = 'lexical performance (average over utterance) '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(1,2, 1)  
plt.plot(m7base1[0], label='layer1')
plt.plot(m7base1[1], label='layer2')
plt.plot(m7base1[2], label='layer3')
plt.plot(m7base1[3], label='layer4')
plt.plot(m7base1[4], label='layer5')
plt.plot(m7base1[5], label='layer6')
plt.plot(m7base1[6], label='layer7')
plt.plot(m7base1[7], label='layer8')
plt.title('base 1: w2v2',size=14)  
plt.xticks([1,2,3,4,5,6],['1', '2', '3', '4', '5','70'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(1,2, 2)  
plt.plot(m7base2[0], label='layer1')
plt.plot(m7base2[1], label='layer2')
plt.plot(m7base2[2], label='layer3')
plt.plot(m7base2[3], label='layer4')
plt.plot(m7base2[4], label='layer5')
plt.plot(m7base2[5], label='layer6')
plt.plot(m7base2[6], label='layer7')
plt.plot(m7base2[7], label='layer8')
plt.title('base 2: VGS ',size=14)  
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],['1', '2', '3', '4', '5','10','20','30','40','50','60','70'])
plt.grid()
plt.legend(fontsize=14) 


 
plt.savefig(os.path.join(path_save, 'lexical_cls_sub3' + '.png'), format='png')