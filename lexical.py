
import torch
torch.cuda.empty_cache()
#%%
# Author: David Harwath
import argparse
import os
import numpy as np
import soundfile as sf
from models.w2v2_model import Wav2Vec2Model_cls 

from steps import trainer
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset

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

# my custom args
parser.add_argument("--mytwd", help="my model dir")
parser.add_argument("--mytarget_layer", help="my target layer")

args = parser.parse_args()

#%% args from script

# root = "/worktmp2/hxkhkh/current/"

# #..............................................................................
# data_root = os.path.join(root, 'FaST/data')
# fb_w2v2_weights_fn = os.path.join(root,'FaST/model/wav2vec_small.pt')
# libri_fn_root = os.path.join(root,'FaST/datavf/libri_fn_root/')
# pretrained_root = os.path.join(root,'FaST/hubertAndDINO')
# #..............................................................................
# args.data_root=data_root
# args.fb_w2v2_weights_fn=fb_w2v2_weights_fn
# args.libri_fn_root=libri_fn_root
# args.load_pretrained_vit=pretrained_root
    
args.batch_size= 4
args.val_batch_size= 16
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

#os.makedirs(args.exp_dir, exist_ok=True)
args.places = False
args.flickr8k = False
args.validate = True

# my args
mytwd = args.mytwd
args.layer_use = int(args.mytarget_layer)

# #############################################################################
# ########################################### defining the model based on ARGS
device = 'cpu'
#..............................
conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

# loading Pre-trained weights
bundle = torch.load(mytwd)
conv1_trm1_trm3.carefully_load_state_dict(bundle['dual_encoder'])

#%%
# #############################################################################
                            # extracting embeddings #
# #############################################################################
# Paths for input and output
wav_path = "/worktmp/khorrami/current/semtest/COCO/"#'/scratch/specog/lextest/data/CDI/'#
save_path = "/worktmp/khorrami/current/lextest/embedds/" # '/scratch/specog/lextest/embedds/'
os.makedirs(save_path, exist_ok=True)
###############################################################################
def LoadAudio( path):
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    length_orig = len(x)
    audio_length = length_orig
    x_norm = (x - np.mean(x)) / np.std(x)    
    return x_norm, audio_length
###############################################################################
# changing device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

wav_files_json = os.listdir(wav_path)

with torch.no_grad():
    for counter, wav_file in enumerate(wav_files_json):
        print(counter)
        signal_peng,l =  LoadAudio(wav_path + wav_file) 
        
        audio_signal = torch.tensor(signal_peng ,dtype=torch.float).to(device)
        input_signal = audio_signal.view(1, -1)
        trm13_out = conv1_trm1_trm3(input_signal,  mask=False, features_only=True, tgt_layer=args.layer_use)
        trm13_out_features = trm13_out['layer_feats']
        output_tensor = trm13_out_features[0] # (time, 768)
        output_np_arr = output_tensor.cpu().detach().np()
        #print(len(output_np_arr))
        output_np_arr_cls = np.mean(output_np_arr, axis = 0)
        # below line is for saving the results as the first row in text file
        output_np_arr_cls = output_np_arr_cls.reshape(1, output_np_arr_cls.shape[0])
        np.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr_cls)
        
        torch.cuda.empty_cache()
       