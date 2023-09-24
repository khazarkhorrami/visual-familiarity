
# for data
import argparse
import soundfile as sf
import numpy as np
import time
import torch
import numpy
import os
import json
from PIL import Image
from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model

from datasets import spokencoco_dataset, libri_dataset
import torchvision.transforms as transforms
#%%

# loading Model
device = 'cpu'
# adding all args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mytwd", help="bundle file dir")
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
mytwd = '/worktmp2/hxkhkh/current/FaST/experiments/vfbase3/bundle.pth'
#mytwd = args.mytwd
# fixed args
args.encoder_layers = 12
args.trim_mask = True
args.normalize = True
args.encoder_attention_heads = 12
args.layer_use = 7
############################################## defining the model based on ARGS

dual_encoder = fast_vgs.DualEncoder(args)
dual_encoder.to(device)
dual_encoder.eval()

# loading Pre-trained weights
bundle = torch.load(mytwd)
dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])

#changing device to gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dual_encoder.to(device)
dual_encoder.eval()

#%% below is related to reading test data
#%%
save_path = '../../semtest/Smatrix/'
file_json_pairings =  "../../semtest/semtest_files_pairings.json"  

audio_path = '../../semtest/COCO/'
image_path = '../../semtest/images/original/'
#%% reading test datafile names 
with open(file_json_pairings, 'r', encoding='utf-8') as json_file:
    data_pairings = json.load(json_file) 

wav_files = []
img_files = []
for key, value in data_pairings.items():
    wav_files.append(os.path.join(audio_path, key))
    img_files.append(os.path.join(image_path, value))
#%%
def LoadAudio(path):
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    # x, sr = librosa.load(path, dtype = 'float32', sr = 16000)
    x_norm = (x - np.mean(x)) / np.std(x)     
    return x_norm, len(x)

def LoadImage(path):
    img = Image.open(path).convert('RGB')
    image_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = image_transform(img)
    return img

#%%

def find_audio_token (signal, l):
    audio_signal = torch.tensor(signal ,dtype=torch.float).to(device)
    input_signal = audio_signal.view(1, -1)
    
    col_audio = torch.nn.utils.rnn.pad_sequence(input_signal, batch_first=True)
    col_al = torch.LongTensor(l).view(1, -1)
    
    audio_attention_mask = torch.arange(len(col_audio[0])).unsqueeze(0) >= col_al.unsqueeze(1)
    audio_feats, cls_token_coarse, extended_audio_attention_mask, losses = dual_encoder.forward_audio(audio_feats = input_signal, audio_attention_mask=audio_attention_mask, test = True)  
    audio_cls = audio_cls_tensor[0] # (1, 768)
    audio_cls_np = audio_cls.cpu().detach().numpy()
    return cls_token_coarse, audio_cls_np

def find_visual_token (signal):
    visual_signal = torch.tensor(signal ,dtype=torch.float).to(device)
    input_signal = visual_signal.view(1, visual_signal.shape[0],  visual_signal.shape[1], visual_signal.shape[2])
    cls_token_coarse = dual_encoder.forward_image(images = input_signal)['visual_cls']#['visual_feats'].mean(1)
    visual_cls = cls_token_coarse[0] # (1, 768)
    visual_cls_np = visual_cls.cpu().detach().numpy()
    return cls_token_coarse, visual_cls_np

#%% For visual    
# There are 1600 jpg files from MSCOCO (masked/blurred)
visual_cls_list = []
start = time.time()    
with torch.no_grad():
    for counter, img_file in enumerate(img_files):
        print(counter)   
        #########################
        signal =  LoadImage(img_file) # (Array of float32) 
        visual_cls_tensor, visual_cls_np = find_visual_token (signal) #(768,) (Array of float32)
        visual_cls_list.append(visual_cls_tensor) 
        

end = time.time()
time_visual = end - start
print(end - start) 
#%%

#%% For audio
# There are 1600 wav files for COCO
audio_cls_list = []
start = time.time()
with torch.no_grad():
    for counter, wav_file in enumerate(wav_files):
        print(counter)   
        #########################
        signal,l =  LoadAudio(wav_file) # (Array of float32) 
        audio_cls_tensor, audio_cls_np = find_audio_token (signal, l) #(768,) (Array of float32)
        audio_cls_list.append(audio_cls_tensor)

end = time.time()
time_audio = end - start
print(end - start)        

       
#%% matchmap

start = time.time()


visual_cls_list = torch.cat(visual_cls_list)
audio_cls_list = torch.cat(audio_cls_list)

matchmap = visual_cls_list @ audio_cls_list.transpose(0,1)
matchmap_np = matchmap.cpu().detach().numpy()

end = time.time()
time_matchmap = end - start
print(end - start) 

np.save( os.path.join(save_path, "Sbest") , matchmap_np)

#%%
cos = torch.nn.CosineSimilarity(dim=1)
s = cos(visual_cls_list, audio_cls_list)
print(s)