#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N adlis
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/adlis
source /beegfs/data/work/imvia/in156281/adlis/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/adlis/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python Classification_resnet18/train.py