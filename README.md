# ADLIS


## Version des paquets

Python 3.9
torch 2.6.0
torchvision 0.21.0
scikit-learn learn 1.6.1
scipy 1.15.0
matplotlib 3.10.1


## Manip CCUB

```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/adlis/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/adlis/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list
```