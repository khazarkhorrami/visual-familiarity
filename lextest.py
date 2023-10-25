root = "/worktmp2/hxkhkh/current/lextest/"
dtype = 'CDI'
mtype = 'DINO'


#%%
import os
import numpy as np
from matplotlib import pyplot as plt
import os

def read_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

def read_all_layers (path, model_name, tn):
    scores = []
    layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8', 'L9', 'L10', 'L11']
    for layer_name in layer_names:
        name = layer_name
        path_full = os.path.join(path , model_name,  name , tn)
        s = read_score (path_full)
        scores.append(s)
    bs = np.sort(scores)[::-1][0]
    bl = np.argsort(scores)[::-1][0]
    return scores, bs, bl

def get_best_score(path, models, tn):
    dict_scores = {}
    dict_bs = {}
    dict_bl = {}
    
    for model in models:
        scores, bs, bl = read_all_layers (path, model, tn)
        dict_scores[model] = scores
        dict_bs[model] = bs
        dict_bl[model] = bl
    return dict_scores, dict_bs, dict_bl

def plotbar_single (names, results , title, cl,logmel):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results)
    
    br1 = np.arange(n)

    ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='black', label='log-Mel features', linewidth=1.5)
    plt.bar(br1, results, color ='b', width = barWidth,
            edgecolor ='grey')
    plt.ylabel('lexical test' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    plt.ylim(0,1) 
    plt.legend(fontsize = 26, framealpha=0.1)
    plt.grid()
    plt.title(title + '\n', fontweight ='bold', fontsize = 26)
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + 'lex' + title + '.png' ,  format = 'png' )
    plt.show()
    
def plotbar_multi_all (names, results , title, cl, logmel):
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 12))  
    
    n = len(results[0])
    
    br1 = np.arange(n)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    x = [-barWidth]
    x.extend(list(br1))
    x.extend([float (list(br1)[-1] + 3*barWidth )])
   
    ychance = np.ones(len(br1))*cl
    ylogmel = np.ones(len(br1))*logmel
    plt.plot(br1,ychance, color ='red', label='chance level', linewidth=1.5)
    plt.plot(br1,ylogmel, color ='black', label='log-Mel features', linewidth=1.5)
    
    plt.bar(br1, results[0], color ='b', width = barWidth,
            edgecolor ='grey', label =title[0])
    plt.bar(br2, results[1], color ='g', width = barWidth,
            edgecolor ='grey', label =title[1])
    plt.bar(br3, results[2], color ='grey', width = barWidth,
            edgecolor ='grey', label =title[2])
    plt.title("comparison between all pretrained versions" + ' (' + dtype + ')' , fontweight ='bold', fontsize = 28)
    plt.ylabel('lexical test' + '\n', fontweight ='bold', fontsize = 28)
    plt.xticks([r + barWidth for r in range(n)], names, fontweight ='bold',fontsize = 26)
    plt.yticks(fontsize = 26)
    #plt.ylim(0,1) 
    plt.legend(fontsize = 24)
    plt.grid()
    savepath = os.path.join(root, "results/" )
    plt.savefig(savepath + 'lexall' + '_' + dtype + '_' + mtype + '.png' ,  format = 'png' )
    plt.show()
#%%    


if dtype=='COCO':
    tn = 'lextest_overall_score.txt'
    cl = 1/80
    logmel = 0.20
else:
    tn = 'output.txt'
    cl = 1/89
    logmel = 0.17
    
if mtype=='RCNN' or mtype=='DINO':
    models = ['expS1', 'expS0', 'expS2', 'expS3']
    names = [ "2\n months", "4\n months\n (uniform)", "4\n months","6\n months"]
elif mtype=='SSL':
    models = ['exphh']
    names = ["SSL 6M"]
    
#%% FB, single

exp = 'expFB'
expname = "FB"
title = mtype + '-' + dtype + '-' + expname
path_input = os.path.join(root, 'output/', dtype, mtype, exp )
dict_scores, dict_bs, dict_bl = get_best_score(path_input, models, tn)
x_FB = []
results_FB = []
for key, value in dict_bs.items():
    x_FB.append(key)
    results_FB.append(value)

plotbar_single (names, results_FB, title, cl, logmel)    
# dict_scores, dict_bs, dict_bl = get_best_score(models, tn)
# # plotting
# x = []
# results = []
# for key, value in dict_bs.items():
#     x.append(key)
#     results.append(value)


# plotbar_single (names, results, title, cl, logmel)

#%% all pretrained models
kh
results = []
title = []

exp = 'expFB'
expname = "FB"
title.append('Pre' + expname)
path_input = os.path.join(root, 'output/', dtype, mtype, exp )
dict_scores, dict_bs, dict_bl = get_best_score(path_input, models, tn)
x_FB = []
results_FB = []
for key, value in dict_bs.items():
    x_FB.append(key)
    results_FB.append(value)
    
exp = 'exp6M'
expname = "6M"
title.append('Pre' + expname)
path_input = os.path.join(root, 'output/', dtype, mtype, exp )
dict_scores, dict_bs, dict_bl = get_best_score(path_input, models, tn)
x_6M = []
results_6M = []
for key, value in dict_bs.items():
    x_6M.append(key)
    results_6M.append(value)
 
exp = 'expR'
expname = "R"
title.append('Pre' + expname)
path_input = os.path.join(root, 'output/', dtype, mtype, exp )
dict_scores, dict_bs, dict_bl = get_best_score(path_input,models, tn)
x_R = []
results_R = []
for key, value in dict_bs.items():
    x_R.append(key)
    results_R.append(value)

results.append(results_FB)
results.append(results_6M) 
results.append(results_R) 

plotbar_multi_all (names, results , title,cl, logmel)