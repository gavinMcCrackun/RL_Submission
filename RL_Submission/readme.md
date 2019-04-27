# RL_Project_2.0
```bash
git clone https://github.com/gavinMcCrackun/RL_Project_2.0
cd RL_Project_2.0
# optional virtualenv
python -m venv env
source env/bin/activate
# install requirements
pip install -r requirements.txt
```
To run
```bash
python -m project.bin.learn ./actor-critic-line.json
```
Note that python looks for the data file specified in the json relative to 
wherever you run from. If the data file `movements.npz` is in `RL_Project_2.0/scratch/`,
e.g. directory looks like
```
RL_Project_2.0
├── .git
├── .gitignore
├── project
├── readme.md
├── requirements.txt
├── actor-critic-line.json
└── scratch
    └── movements.npz
```
you'd do something like:
```bash
cd scratch
PYTHONPATH=$PYTHONPATH:../ python -m project.bin.learn ../actor-critic-line.json
# or for the stats entry point (does nothing but load movements)
PYTHONPATH=$PYTHONPATH:../ python -m project.bin.stats ./movements.npz
```
