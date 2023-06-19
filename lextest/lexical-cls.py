import os
#############################################################################
#twd = e.g., '../experiments/model19base3/9252_bundle.pth'
#target_layer = e.g., 2
total_layers = 12
trimTF = True

# Paths for data, select testing on COCO/CDI
wav_path_COCO = '/scratch/specog/lextest/data/COCO/'
wav_path_CDI = '/worktmp2/hxkhkh/current/lextest/data/CDI/'
wav_path = wav_path_COCO

# Path for saving embeddings
save_path = '/scratch/specog/lextest/embedds/'
os.makedirs(save_path, exist_ok=True)
#############################################################################
# for data
import argparse
import soundfile as sf
import numpy as np
import torch
import numpy
# for model
from models.w2v2_model import  Wav2Vec2Model_cls , ConvFeatureExtractionModel
from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset

#%% writting resamples files, 
# run this section only once if you want to save resampled data and 
# use sf for reading them (sf is faster than librosa)

# import librosa
# import os 
# import soundfile as sf
# import time
# t1 = time.time()

# wav_path = '/worktmp2/hxkhkh/current/lextest/COCO_lextest/COCO_synth/'
# new_path = '/worktmp2/hxkhkh/current/lextest/data/COCO/'

# SR = 16000
# wav_files = os.listdir(wav_path)

# for counter, wav_file in enumerate(wav_files):
    
#     x, sr = librosa.load(wav_path + wav_file, dtype = 'float32', sr = SR)
#     #sf.write(new_path + wav_file , x, SR)
    
# t2 = time.time()
# print(t2-t1)
#%%

def LoadAudio( path):
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    # x, sr = librosa.load(path, dtype = 'float32', sr = 16000)
    x_norm = (x - np.mean(x)) / np.std(x)     
    return x_norm, len(x)

#%%

# loading Model
device = 'cpu'
# adding all args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mytwd", help="my model dir")
parser.add_argument("--mytarget_layer", help="my target layer")
#..............................................................................

parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
parser.add_argument("--test", action="store_true", default=False, help="test the model on test set")
trainer.Trainer.add_args(parser)
w2v2_model.Wav2Vec2Model_cls.add_args(parser)
fast_vgs.DualEncoder.add_args(parser)
spokencoco_dataset.ImageCaptionDataset.add_args(parser)
libri_dataset.LibriDataset.add_args(parser)
args = parser.parse_args()
#..............................

# input args
mytwd = args.mytwd 
args.layer_use = int(args.mytarget_layer)


# fixed args
args.encoder_layers = total_layers
args.trim_mask = trimTF
args.normalize = True
args.encoder_attention_heads = 12

print ('###############################')
print(args)
print ('###############################')

############################################## defining the model based on ARGS
#..............................
conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()


# loading Pre-trained weights

bundle = torch.load(mytwd)
conv1_trm1_trm3.carefully_load_state_dict(bundle['dual_encoder'])

#############################################################################

# changing device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

wav_files = os.listdir(wav_path)

with torch.no_grad():
    for counter, wav_file in enumerate(wav_files):
        print(counter)
        signal_peng,l =  LoadAudio(wav_path + wav_file) 
        
        audio_signal = torch.tensor(signal_peng ,dtype=torch.float).to(device)
        input_signal = audio_signal.view(1, -1)
        trm13_out = conv1_trm1_trm3(input_signal,  mask=False, features_only=True, tgt_layer=args.layer_use)
        trm13_out_features = trm13_out['layer_feats']
        output_tensor = trm13_out_features[0] # (time, 768)
        output_np_arr = output_tensor.cpu().detach().numpy()
        #print(len(output_np_arr))
        output_np_arr_cls = numpy.mean(output_np_arr, axis = 0)
        # below line is for saving the results as the first row in text file
        output_np_arr_cls = output_np_arr_cls.reshape(1, output_np_arr_cls.shape[0])
        numpy.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr_cls)
        
        torch.cuda.empty_cache()
        #del trm13_out,trm13_out_features,output_tensor,output_np_arr

############################################################################
# from matplotlib import pyplot as plt
# plt.imshow(output_np_arr.T)
# vec = {'embedding_pretrained_model':output_np_arr}
# from scipy.io import savemat
# savemat('/home/hxkhkh/Music/' + "embedding_w2v2_model_layer3.mat", vec)
