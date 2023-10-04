#!/bin/sh
DATAFOLDER='/worktmp/khorrami/current/semtest/COCO/'
EMBDFOLDER='/worktmp/khorrami/current/lextest/embedds/'
NAME="expS3"
OUTFOLDER="/worktmp/khorrami/current/lextest/output/COCO"/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments/vfsubsets/prefb"/$NAME

source activate fastvgs

M="best_bundle.pth"
for LAYERNAME in 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done

