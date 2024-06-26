#!/bin/sh
export PATH="/projappl/project_2001315/khazar/con/vf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
fb_w2v2_weights_fn="../../../../model/wav2vec_small.pt"
exp_dir="../../expls10/"
twd="../../twd/"
libri_fn_root="../../../../datavf/libri_fn_root/"
pretrained_root="../../../../hubertAndDINO"

python \
../run_spokencoco.py \
--ssl \
--data_root ${data_root} \
--trained_weights_dir ${twd} \
--exp_dir ${exp_dir} \
--libri_fn_root ${libri_fn_root} \
--load_pretrained_vit ${pretrained_root} \
--batch_size 24 \
--val_batch_size 24 \
--val_cross_batch_size 22 \
--n_epochs 200 \
--n_print_steps 100 \
--n_val_steps 1800 \
--lr 0.0005 \
--warmup_fraction 0.08 \
--vit_arch 'vitsmall' \
--vit_patch_size 8 \
--vit_checkpoint_key 'teacher' \
--xtrm_layers 1 \
--trm_layers 1 \
--fine_matching_weight 0.0 \
--coarse_matching_weight 1.0 \
--libri_w2v2_weight 0.0 \
--caption_w2v2_weight 1.0 \
--feature_grad_mult 0.1 \
--trim_mask \
--encoder_layers 10 \
--encoder_attention_heads 8 \
--layer_use 7 \

