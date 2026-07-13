#!/bin/ksh 
#$ -q gpu
#$ -o detection_result.out
#$ -j y
#$ -N adlis_detection
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/adlis
source /beegfs/data/work/imvia/in156281/adlis/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/adlis/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export TORCH_HOME=/work/imvia/in156281/adlis/.cache/torch
python detection/train.py