#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfsubsets/prefb"


MNAME="expS1"
SNAME="S1_aL_vM"
AF="COCO"
VF="images/masked"
python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS2"
SNAME="S2_aL_vM"
AF="COCO"
VF="images/masked"
python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS3"
SNAME="S3_aL_vM"
AF="COCO"
VF="images/masked"
python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF


