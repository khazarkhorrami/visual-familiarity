#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfsubsets/pre6M"

MNAME="expS3"
SNAME="S3_aL_vM"
AF="COCO"
VF="images/masked"
python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF


