



#%%
# Author: David Harwath
import argparse
import os
import numpy as np
import pickle
import time
from steps import trainer
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset
from logging import getLogger
import logging


logger = getLogger(__name__)
# khazar added below ....
logger.setLevel(logging.DEBUG)
logging.basicConfig()
# .......................

logger.info("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
parser.add_argument("--test", action="store_true", default=False, help="test the model on test set")

parser.add_argument("--ssl", action="store_true", dest="ssl", help="only ssl training")

trainer.Trainer.add_args(parser)

w2v2_model.Wav2Vec2Model_cls.add_args(parser)

fast_vgs.DualEncoder.add_args(parser)

spokencoco_dataset.ImageCaptionDataset.add_args(parser)

libri_dataset.LibriDataset.add_args(parser)

args = parser.parse_args()

#%% args from script

#exp_dir = '/worktmp2/hxkhkh/current/FaST/experiments/vfbase3/exp/'
exp_dir = '/worktmp2/hxkhkh/current/FaST/experiments/vfsubsets/expS3/'
data_root = '/worktmp2/hxkhkh/current/FaST/data'
fb_w2v2_weights_fn = '/worktmp2/hxkhkh/current/FaST/model/wav2vec_small.pt'
libri_fn_root = '/worktmp2/hxkhkh/current/FaST/datavf/libri_fn_root/'
pretrained_root = '/worktmp2/hxkhkh/current/FaST/hubertAndDINO'

args.data_root=data_root
args.fb_w2v2_weights_fn=fb_w2v2_weights_fn
args.exp_dir=exp_dir
args.libri_fn_root=libri_fn_root
args.load_pretrained_vit=pretrained_root
    
args.batch_size= 4
args.val_batch_size= 8
args.val_cross_batch_size= 4
args.n_epochs= 50
args.n_print_steps= 100
args.n_val_steps= 1000
args.lr= 0.0001
args.warmup_fraction= 0.1
args.vit_arch= 'vitsmall'
args.vit_patch_size= 8
args.vit_checkpoint_key= 'teacher'
args.normalize= True
args.xtrm_layers= 1
args.trm_layers= 6
args.fine_matching_weight= 0.0
args.coarse_matching_weight= 1.0
args.libri_w2v2_weight= 0.0
args.caption_w2v2_weight= 1.0
args.feature_grad_mult= 1.0
args.trim_mask= True
args.layer_use= 7
    
#%%

os.makedirs(args.exp_dir, exist_ok=True)

if args.resume or args.validate:
    resume = args.resume
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        old_args = pickle.load(f)
    new_args = vars(args)
    old_args = vars(old_args)
    for key in new_args:
        if key not in old_args or old_args[key] != new_args[key]:
            old_args[key] = new_args[key]
    args = argparse.Namespace(**old_args)
    args.resume = resume
else:
    print("\nexp_dir: %s" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
args.places = False
args.flickr8k = False
args.validate = True
args.test = True
#%%
 
device = 'cpu'
my_trainer = trainer.Trainer(args)
batch, s = my_trainer.validate_khazar()



audio = batch['audio'].cpu().detach().numpy()
atm = batch['audio_attention_mask'].cpu().detach().numpy()
al = batch['audio_length'].cpu().detach().numpy()
images = batch ['images'].cpu().detach().numpy()
img_id = batch ['img_id']
fn = batch['fn']

s_np = s.cpu().detach().numpy()

save_path = "/worktmp2/hxkhkh/current/semtest/Smatrix"
np.save( os.path.join(save_path, "S3_words_masked") , s_np)

#%%
kh
#%% below is related to reading test data
import json
import os
audio_dataset_json_file = os.path.join(args.data_root, "coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
with open(audio_dataset_json_file, 'r', encoding='utf-8') as json_file:
    data_json = json.load(json_file) 
data = data_json['data']
# image, caption (text, speaker, uttid, wav)
img_example = data[0]['image']
wav_example = data[0]['caption']['wav']
print(img_example)
print(wav_example)
#val2014/COCO_val2014_000000325114.jpg
#wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav

datum = data[0]#[index]
#img_id = datum['image'].split("/")[-1].split(".")[0] ----> 'COCO_val2014_000000325114'

#wavpath = os.path.join(self.audio_base_path, datum['caption']['wav'])
#imgpath = os.path.join(self.image_base_path, datum['image'])

# audio, nframes = self._LoadAudio(wavpath)
# img = self._LoadImage(imgpath)

# self.audio_base_path = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO"
# self.image_base_path = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/MSCOCO"
# for semtest:
# audio_base_path = "/worktmp2/hxkhkh/current/semtest/utterances/"
# image_base_path = "/worktmp2/hxkhkh/current/semtest/images/original/"

#%%
save_path = '/worktmp2/hxkhkh/current/semtest/'
file_json_pairings =  "../../semtest/semtest_files_pairings.json"  
# reading test datafile names 
with open(file_json_pairings, 'r', encoding='utf-8') as json_file:
    data_pairings = json.load(json_file) 

wav_files = []
img_files = []
data_list = []
for w, i in data_pairings.items():
    wav_files.append(w)
    img_files.append(i)
    d = {}
    d['image'] = i
    d['caption'] = {} 
    d['caption']['wav'] = w
    data_list.append(d)

data_list[0]['caption']['wav']
data_dict = {}
data_dict['data'] = data_list 

#%%
file_json = "/worktmp2/hxkhkh/current/semtest/data.json"
with open(file_json, "w") as fp:
    json.dump(data_dict,fp)
#%% 
