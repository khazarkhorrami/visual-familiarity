# post-processing: audio duration and counts for each subset
import os
import json
import soundfile as sf
import csv

path_audio = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO"
path_meta = "/worktmp2/hxkhkh/current/FaST/datavf/coco_pyp/subsets"
p1 = os.path.join(path_meta, "SpokenCOCO_train_subset1.json")
p2 = os.path.join(path_meta, "SpokenCOCO_train_subset2.json")
p3 = os.path.join(path_meta, "SpokenCOCO_train_subset3.json")
p0 = os.path.join(path_meta, "SpokenCOCO_train_subset0A.json")
pssl = "/worktmp2/hxkhkh/current/FaST/datavf/ssl6M_root/train.tsv"
root = "/worktmp2/hxkhkh/current/FaST/data"

########################################################################### SSL
p = [p1,p2,p3,p0]
with open(pssl, 'r', encoding='utf-8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    names = [row[0] for row in reader]

dur_ssl_all = 0
dur_ssl_SC = 0
dur_ssl_LS = 0
for n in names:
    af = os.path.join(root, n)
    x, sr = sf.read(af, dtype = 'float32')
    l = len(x)/sr
    dur_ssl_all += l
    if n.startswith("LS"):
        dur_ssl_LS += l
    elif n.startswith('coco'):
        dur_ssl_SC+= l
print(round (dur_ssl_all/ 3600, 3)   )
print(round (dur_ssl_SC/ 3600, 3)   ) # 446.36
print(round (dur_ssl_LS/ 3600, 3)   ) # 602.639


count_SC = 0
count_LS = 0
for n in names: 
    if n.startswith("LS"):
        count_LS += 1
    elif n.startswith('coco'):
        count_SC += 1

print(count_SC)
print(count_LS)

########################################################################### VGS
p = [p1,p2,p3,p0]
durations_subsets = []
for pi in p:
    with open(pi, "r") as f:
        m = json.load(f)['data']
        print(len(m))
    l_all = 0
    for mi in m:  
        wf = mi['caption']['wav']
        af = os.path.join(path_audio, wf)
        x, sr = sf.read(af, dtype = 'float32')
        l = len(x)/sr
        l_all += l
    durations_subsets.append(round(l_all/3600 , 3))

# durations_subsets = [1.544, 3.075, 4.613, 3.312 ] 


