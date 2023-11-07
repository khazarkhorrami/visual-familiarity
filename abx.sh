#!/bin/sh
DATAFOLDER='/worktmp/khorrami/current/ZeroSpeech/data/phonetic/'
EMBDFOLDER='/worktmp/khorrami/current/ZeroSpeech/submission'
TYPE='phonetic/'
NAME="vfls"
M="best_bundle.pth"
OUTDIR="/worktmp/khorrami/current/ZeroSpeech/output/WC"/$NAME
mkdir $OUTDIR

MODEL='expnewl8'
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$NAME/$MODEL

OUTFOLDER=$OUTDIR/$MODEL/"best"
mkdir $OUTDIR/$MODEL
mkdir $OUTFOLDER

OUTNAME="L5"
OUTFILE=$OUTFOLDER/$OUTNAME
source activate fastvgs
python /worktmp/khorrami/current/FaST/visual-familiarity/abx.py --apath $DATAFOLDER --epath $EMBDFOLDER/$TYPE --mytarget_layer 0 --mytwd $MFOLDER/$M
conda activate zerospeech2021
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  $EMBDFOLDER -o $OUTFILE -j12 --no-lexical --no-syntactic --no-semantic --force-cpu 
rm -r $EMBDFOLDER/$TYPE

for LAYERNAME in 1 6
do
    OUTNAME="L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    conda activate fastvgs
    python /worktmp/khorrami/current/FaST/visual-familiarity/abx.py --apath $DATAFOLDER --epath $EMBDFOLDER/$TYPE --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    conda activate zerospeech2021
    mkdir $OUTFILE
    zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  $EMBDFOLDER -o $OUTFILE -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
    rm -r $EMBDFOLDER/$TYPE
done

