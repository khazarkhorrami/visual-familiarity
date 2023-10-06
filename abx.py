import torch
torch.cuda.empty_cache()
#%%
import argparse
import os
import numpy as np
import json
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
parser.add_argument("--apath", help="test audio wav dir")
parser.add_argument("--epath", help="path to dave embeddings")
parser.add_argument("--mytwd", help="my model dir")
parser.add_argument("--mytarget_layer", help="my target layer")
args = parser.parse_args()

#%% args from script
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
wav_path = args.apath
save_path = args.epath
os.makedirs(save_path, exist_ok=True)

audio_dataset_json_file = os.path.join(wav_path ,'index.json')

with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
    
d = data_json['subsets']['dev_clean']
wav_files_json = d['items']['wav_list']['files_list']
###############################################################################
def LoadAudio( path):
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    length_orig = len(x)
    audio_length = length_orig
    x_norm = (x - np.mean(x)) / np.std(x)    
    return x_norm, audio_length

#############################################################################
# changing device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

with torch.no_grad():
    for counter, wav_file in enumerate(wav_files_json):
        print(counter)
        signal_peng,l =  LoadAudio(wav_path + wav_file)
        
        audio_signal = torch.tensor(signal_peng ,dtype=torch.float).to(device)
        input_signal = audio_signal.view(1, -1)
        trm13_out = conv1_trm1_trm3(input_signal,  mask=False, features_only=True, tgt_layer=args.layer_use)
        trm13_out_features = trm13_out['layer_feats']
        output_tensor = trm13_out_features[0] # (time, 768)
        output_np_arr = output_tensor.cpu().detach().numpy()
        np.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr )

        torch.cuda.empty_cache()
        del trm13_out,trm13_out_features,output_tensor,output_np_arr

############################################################################
# from matplotlib import pyplot as plt
# plt.imshow(output_np_arr.T)
# vec = {'embedding_pretrained_model':output_np_arr}
# from scipy.io import savemat
# savemat('/home/hxkhkh/Music/' + "embedding_w2v2_model_layer3.mat", vec)
