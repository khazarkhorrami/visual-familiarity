
import json

caption_json = "/worktmp2/hxkhkh/current/FaST/data/coco_subsets/captions_train2014_subset1.json"
with open(caption_json, 'r') as fp:
    data_json_test = json.load(fp)
    
    
audio_dataset_json_file = '/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']
