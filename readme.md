# DeepONet for the Wave Equation

This repository contains the code for the final project at the Scientific Machine Learning course at DTU for Sándor Földi and Sigurd Thorlund.

In this project, we train a DeepONet architecture to solve the wave equation given some initial condition as sensed by a fixed number of sensors, and inputs $x$ and $t$.

Project structure:
```
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
runs                           # outputs of the training 
└── ...
sessions
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
