#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfsubsets"


MNAME="expS1"
SNAME="S1_aL_vB"
AF="COCO"
VF="images/blurred"

python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF
