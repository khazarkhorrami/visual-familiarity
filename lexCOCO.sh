#!/bin/sh
DATAFOLDER=''/worktmp/khorrami/current/lextest/data/COCO/'
EMBDFOLDER='/worktmp/khorrami/current/lextest/embedds/'
NAME="expS1"
OUTFOLDER="/worktmp/khorrami/current/lextest/output/vfsubsets/COCO"/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments/vfsubsets/exp100"/$NAME

source activate fastvgs

M="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done

