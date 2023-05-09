#!/bin/sh
#BSUB -J TrainHpcDefault
#BSUB -W 6:00
#BSUB -q gpuv100
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo
module load python3/3.9.11
source ../env/bin/activate

echo "Running script..."
python3 ./src/train.py dataset=1000 model=deeponet_dense
