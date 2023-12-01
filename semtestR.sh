#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfrand/exprandom/"

mkdir $ROOT/"semtest/S/baseline/"


SNAME="baseline/S_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME="baseline/S_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME="baseline/S_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

