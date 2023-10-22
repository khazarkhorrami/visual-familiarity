#!/bin/sh
DATAFOLDER='/worktmp/khorrami/current/lextest/data/COCO/'
EMBDFOLDER='/worktmp/khorrami/current/lextest/embedds/'
OUTDIR='/worktmp/khorrami/current/lextest/output/COCO'
source activate fastvgs

M="RCNN"
G="expR"
OUTPATH=$OUTDIR/$M/$G
mkdir $OUTPATH


NAME="expS0"
OUTFOLDER=$OUTPATH/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$M/$G/$NAME
B="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$B
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done


NAME="expS1"
OUTFOLDER=$OUTPATH/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$M/$G/$NAME
B="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$B
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done

NAME="expS2"
OUTFOLDER=$OUTPATH/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$M/$G/$NAME
B="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$B
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done

NAME="expS3"
OUTFOLDER=$OUTPATH/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$M/$G/$NAME
B="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$B
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/COCO_lextest
    sh COCO_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done
