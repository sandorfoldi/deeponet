# DeepONet for the Wave Equation

This repository contains the code for the final project at the Scientific Machine Learning course at DTU for Sándor Földi and Sigurd Thorlund.

In this project, we train a DeepONet architecture to solve the wave equation given some initial condition as sensed by a fixed number of sensors, and inputs $x$ and $t$.

Project structure:
```
bash                            # bash scripts, e.g. for training on dtu hpc 
└── train_hpc_default.sh
data                            # data 
├── 10
│   ├── 0.npy
│   ├── 1.npy
│   ├── ...
│   └── 9.npy
├── 100
│   ├── 0.npy
│   ├── ...
│   └── 99.npy
└── 1000
    └── ...
env                            # virtual environment created by get_started.sh
├── bin
│   └── activate
└── ...
notebooks                      # jupyter notebooks
└── ...
runs                           # outputs of the training sessions
└── ...
src                            # codebase
├── viz.py                     # vizualizations
├── wave_generator.py          # script that generates data for the wave equation 
└── wave_loader.py             # torch.data.Dataset class definition
get_started.sh                 # creates virtual environment, installs packages, generates data
readme.md                      # readme file
requirements.txt               # used to install relevant packages
```

## Installation

To install the required packages, run the following command in the terminal:
```
./get_started.sh
```
if you get a permission error, run the following command:
```
chmod u+x get_started.sh
```


## Running on DTU HPC

The model can be trained on dtu hpc, making use of v100 instances. So far, only single gpu training has been implemented.

To train on DTU HPC, first, install the project by running the get_started.sh script. At the end of this process, you should have the data directory populated with datasets, and a virtual environment, set up in the env directory.

### Default
Once this is completed, you can train the network on a v100 accelerator, by running bash/train_hpc_default.sh with bsub, i.e. typing:
```
bsub bash/train_hpc_default.sh
```
Then, you can check the status of the job you have submitted by typing:
```
bstat
```

### Custom
You can also create new training scripts, by changing some parts of the template. If you think the new scripts will be useful for others too, please save it in a new file and name it thoughtfully:)

You might want to change some of the following options:
  - -J: job name, this will be displayed when you call bstat
  - -W: wall/wait/smth, time allocated for your job. If you make this smaller, your job might get started faster. (Or the other way around, if you ask for a v100 for 24 hours, you might have to wait a bit more to get the job started) 
  - -q: queue that the job will be subscribed to (ours is the v100, pretty default for deep learning, but e.g. there is also an a100 queue)

E.g., you might want to create multiple bash scripts for training a model on different datasets, or training different models, etc. Then, you might want to create multiple train_hpc_model_dataset.sh scripts, where you only modify the job name option ("-J"), so that you can start them at the same time, and see which one has finished, and which hasn't when you run bstat.