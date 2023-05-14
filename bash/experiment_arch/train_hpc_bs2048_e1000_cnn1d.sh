#!/bin/sh
#BSUB -J BS_CNN1D
#BSUB -W 1:00
#BSUB -q gpuv100
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "loading python"
module load python3/3.9.11
source env/bin/activate

echo "Running script..."
python src/model_training.py --dataset data/1000 --epochs 1000 --model CNN1D --batch_size 2048 --outputfolder CNN1D --run_name CNN1D