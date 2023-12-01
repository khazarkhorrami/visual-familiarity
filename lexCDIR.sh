#!/bin/sh
DATAFOLDER='/worktmp/khorrami/current/lextest/data/CDI/'
EMBDFOLDER='/worktmp/khorrami/current/lextest/embedds/'
OUTDIR='/worktmp/khorrami/current/lextest/output/CDI'
source activate fastvgs

OUTFOLDER=$OUTDIR/"baseline"
mkdir $OUTFOLDER

MFOLDER="/worktmp/khorrami/current/FaST/experiments/vfrand/exprandom"
B="bundle.pth"
for LAYERNAME in 0 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /worktmp/khorrami/current/FaST/visual-familiarity/lexical.py --apath $DATAFOLDER --epath $EMBDFOLDER --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$B
    mkdir $OUTFILE
    cd /worktmp/khorrami/current/lextest/CDI_lextest
    sh CDI_lextest.sh $DATAFOLDER $EMBDFOLDER 'single' 0 $OUTFILE
    rm -r $EMBDFOLDER
done
