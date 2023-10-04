#!/bin/sh

source activate fastvgs

ROOT="/worktmp2/hxkhkh/current"
MROOT="/worktmp2/hxkhkh/current/FaST/experiments/vfsubsets/prefb"


MNAME="expS1"
SNAME="S1_test"
AF="COCO"
VF="images/blurred"

python semantic_eval_DINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF
