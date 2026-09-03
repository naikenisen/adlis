#!/bin/ksh 
#$ -q gpu
#$ -o figure3bis.out
#$ -j y
#$ -N figure3bis
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/adlis
source /beegfs/data/work/imvia/in156281/adlis/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/adlis/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export TORCH_HOME=/work/imvia/in156281/adlis/.cache/torch
python figures_scripts/figure3bis.py