# mostly borrowed from fairseq/fairseq/data/audio/hubert_dataset.py
import random
import numpy as np
import os
import torch
import torch.nn.functional
import random
import soundfile as sf
from torch.utils.data import Dataset
import pickle
import itertools
import logging
logger = logging.getLogger(__name__)


# manifest_path = "/worktmp2/hxkhkh/current/FaST/data/LS/libri_fn_root/valid.tsv"
# max_keep = 16000*80
# min_keep = 32000
# kh
def load_audio(manifest_path, max_keep, min_keep):
    # Kh: I added this
    data_path_root_splits = manifest_path.split('/')[0:-2]
    root = '/'.join(data_path_root_splits)
    trash = []
    hours = []
    # Kh
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        # Kh: I chaned this
        #root = f.readline().strip()
        # Kh
        f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
                
            elif max_keep is not None and sz > max_keep:
                n_long += 1
                trash.append(items)
            else:
                #kh: i changed this 
                #names.append(items[0])
                file_splits = items[0].split("/")[-5:]
                file_name = '/'.join(file_splits)
                
                
                names.append(file_name)
                #kh
                
                inds.append(ind)
                sizes.append(sz)
                hours.append((sz/16000)/3600)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    # Kh: I changed root to my own defined root
    # to have file_path = os.path.join(data_root, file_name)
    # Kh
    # return root, names, inds, tot, sizes
    return root, names, inds, tot



# def verify_label_lengths(
#     audio_sizes,
#     audio_rate,
#     label_path,
#     label_rate,
#     inds,
#     tot,
#     tol=0.1,  # tolerance in seconds
# ):
#     if label_rate < 0:
#         logger.info(f"{label_path} is sequence label. skipped")
#         return

#     with open(label_path) as f:
#         lengths = [len(line.rstrip().split()) for line in f]
#         assert len(lengths) == tot
#         lengths = [lengths[i] for i in inds]
#     num_invalid = 0
#     for i, ind in enumerate(inds):
#         dur_from_audio = audio_sizes[i] / audio_rate
#         dur_from_label = lengths[i] / label_rate
#         if abs(dur_from_audio - dur_from_label) > tol:
#             logger.warning(
#                 (
#                     f"audio and label duration differ too much "
#                     f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
#                     f"in line {ind+1} of {label_path}. Check if `label_rate` "
#                     f"is correctly set (currently {label_rate}). "
#                     f"num. of samples = {audio_sizes[i]}; "
#                     f"label length = {lengths[i]}"
#                 )
#             )
#             num_invalid += 1
#     if num_invalid > 0:
#         logger.warning(
#             f"total {num_invalid} (audio, label) pairs with mismatched lengths"
#         )

class LibriDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--libri_fn_root", type=str, default="/data3/scratch/pyp/exp_pyp/libri/", help="from fairseq mae simple kmeans")
        parser.add_argument("--libri_max_seq_len", type=float, default=8)
        parser.add_argument("--libri_val_bzs", type=int, default=4)
        parser.add_argument("--sample_rate", type=int, default=16000)
        parser.add_argument("--feature_rate", type=int, default=50, help="50")
        parser.add_argument("--label_rate", type = int, default=100, help="the number of labels per second of audio. 100 if mfcc, 50 is MAE features")
        parser.add_argument("--feature_dim", type=int,
            default=100,
            help = "dim feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 80"
        )
        parser.add_argument("--deltas", action="store_true",
            default=True,
            help = "whether or not add delta and delta-delta to the feature, only effective for spectrogram/fbank/mfcc"
        )
        parser.add_argument("--feature_type", type=str, default="wav", help="choose from wav/spectrogram/fbank/mfcc")
        parser.add_argument("--max_keep_sample_size", type=int, default=16000*80)
        parser.add_argument("--min_keep_sample_size", type=int, default=32000)
    def __init__(self, args, split):
        self.args = args
        self.split = split

        if "train" in split:
            manifest_path = os.path.join(self.args.libri_fn_root, "train.tsv")
        elif "val" in split or "valid" in split or "dev" in split:
            manifest_path = os.path.join(self.args.libri_fn_root, "valid.tsv")

        self.audio_root, self.audio_names, inds, tot = load_audio(
            manifest_path, self.args.max_keep_sample_size, self.args.min_keep_sample_size
        )

    def __len__(self):
        return len(self.audio_names)

    def calculate_batch_size(self, num_steps):
        print('.........here is calculating .............')
        print('........ len self ........................')
        print(len(self))
        print('........ num steps  ........................')
        print(num_steps)
        print('........ calculated bs ........................')
        print (str (int(np.ceil(len(self)/num_steps))))
        return int(np.ceil(len(self)/num_steps))

    def _LoadAudioLabel(self, fn, label_key):
        #Kh: "label_key" is file name without ".flac" but it is not used here
        x, sr = sf.read(fn, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.args.libri_max_seq_len: 
            audio_length = int(16000 * self.args.libri_max_seq_len)
            
            # Kh: this part is diferent from COCO audio load
            # in COCO for both train and val we have 
            # x = x[:audio_length]
            # ..............................................
            if "train" in self.split:
                start_max = length_orig - audio_length
                start = random.choice(range(start_max))
                x_temp = x[start:(start+audio_length)]
                if np.linalg.norm(x_temp) != 0:
                    x = x_temp
                else:
                    x = x[:audio_length]
            else:
                x = x[:audio_length]
            # Kh ..........................................   
            
            x_norm = (x - np.mean(x)) / np.std(x) # normalize per instance
            x = torch.FloatTensor(x_norm)
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.args.libri_max_seq_len))
            x_norm = (x - np.mean(x)) / np.std(x) # normalize per instance
            new_x[:audio_length] = torch.FloatTensor(x_norm) 
            x = new_x
        return x, audio_length

    def __getitem__(self, index):
        fn = os.path.join(self.audio_root, self.audio_names[index])
        label_key = "/".join(fn.split("/")[-4:]).split(".")[0]
        wav, wav_len = self._LoadAudioLabel(fn, label_key)
        return wav, wav_len, label_key


    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[0], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[1])
        collated['id'] = vals[2]
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        return collated
