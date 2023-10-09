#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfsubsets/exp100"

MNAME="expS0"
SNAME="S0_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS0"
SNAME="S0_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS0"
SNAME="S0_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF
