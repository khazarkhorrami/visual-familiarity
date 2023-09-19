
# for data
import argparse
import soundfile as sf
import numpy as np
import torch
import numpy
import os
from PIL import Image
# for model
#os.chdir('../') # worktmp2/hxkhkh/current/FaST/visual-familiarity
from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model
#from models.fast_vgs import 
from datasets import spokencoco_dataset, libri_dataset

#%%
import torchvision.transforms as transforms

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
save_path = '../../lextest/tokensA/'
#twd = '../exp/bundle.pth'
 
#%%
audio_path = "../../lextest/data/COCO/"
audio_file = "airplane_1.wav"
audio_file_path = os.path.join(audio_path,audio_file)
x, sr = sf.read(audio_file_path, dtype = 'float32')

image_file_path = "/worktmp2/hxkhkh/current/FaST/data/coco_example/subset1/images/1.jpg"
img = LoadImage(image_file_path)
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
#mytwd = '../exp/bundle.pth'
mytwd = args.mytwd
# fixed args
args.encoder_layers = 12
args.trim_mask = True
args.normalize = True
args.encoder_attention_heads = 12

############################################## defining the model based on ARGS
#..............................

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
# There are 1600 wav files for COCO
wav_files = os.listdir(audio_path)


def find_audio_token (signal):
    audio_signal = torch.tensor(signal ,dtype=torch.float).to(device)
    input_signal = audio_signal.view(1, -1)  
    audio_cls = dual_encoder.forward_audio_khazar(audio_feats = input_signal)#, audio_attention_mask=None, test = True)  
    output_audio_tensor = audio_cls[0] # (1, 768)
    audio_cls_np = output_audio_tensor.cpu().detach().numpy()
    return audio_cls_np

def find_visual_token (signal):
    visual_signal = torch.tensor(signal ,dtype=torch.float).to(device)
    input_signal = visual_signal.view(1, visual_signal.shape[0],  visual_signal.shape[1], visual_signal.shape[2])
    visual_cls = dual_encoder.forward_image(images = input_signal)['visual_cls']
    output_visual_tensor = visual_cls[0] # (1, 768)
    visual_cls_np = output_visual_tensor.cpu().detach().numpy()
    return visual_cls_np

#########################

with torch.no_grad():
    signal = LoadImage(image_file_path) # (3, 224, 224) (Tensor object)
    visual_cls_np = find_visual_token (signal) #(768,) (Array of float32)
    print(np.shape(visual_cls_np))
    
# with torch.no_grad():
#     for counter, wav_file in enumerate(wav_files):
#         print(counter)   
#         #########################
#         audio_file_path = os.path.join(audio_path, wav_file)
#         signal,l =  LoadAudio(audio_file_path) # (Array of float32) 
#         audio_cls_np = find_audio_token (signal) #(768,) (Array of float32)
#         #.......................................
#         save_name = wav_file.split('.')[0]
#         save_file = os.path.join(save_path, save_name )
#         np.save(save_file, audio_cls_np)
        
    