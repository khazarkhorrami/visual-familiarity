#!/bin/sh

source activate fastvgs

M="ssl"
G="exp15"
ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments"/$M/$G

mkdir $ROOT/"semtest/S"/$M/$G


SNAME=$M/$G/"S0_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S0_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$M/$G/"S0_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

