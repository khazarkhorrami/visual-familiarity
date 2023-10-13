#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
GNAME="vfsubsets"
MROOT="/worktmp/khorrami/current/FaST/experiments/$GNAME/exp6M"
mkdir $ROOT/"semtest"/"Smatrix"/$GNAME


MNAME="expS0"

SNAME=$GNAME/"S0_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S0_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S0_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF



MNAME="expS1"

SNAME=$GNAME/"S1_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S1_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S1_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

MNAME="expS2"

SNAME=$GNAME/"S2_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S2_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S2_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF


MNAME="expS3"

SNAME=$GNAME/"S3_aL_vO"
AF="COCO"
VF="images/original"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S3_aL_vM"
AF="COCO"
VF="images/masked"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

SNAME=$GNAME/"S3_aL_vB"
AF="COCO"
VF="images/blurred"
python semDINO.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF --vfiles $VF

