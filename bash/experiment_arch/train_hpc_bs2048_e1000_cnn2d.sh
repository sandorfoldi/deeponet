#!/bin/sh
#BSUB -J BS_CNN2D
#BSUB -W 2:00
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
python src/model_training.py --dataset data/1000 --epochs 1000 --model CNN2D --batch_size 2048 --outputfolder CNN2D --run_name CNN2D