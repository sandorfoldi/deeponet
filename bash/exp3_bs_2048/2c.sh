
#!/bin/sh
#BSUB -J exp3_2c
#BSUB -W 12:00
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
python src/train_pinn.py --dataset /work3/s216416/deeponet/data/2c --outputfolder exp3_2c --run_name exp3_2c --epochs 500 --n_hidden 128  --batch_size 2048 --mu_boundary 1 --mu_colloc 0 --mu_ic 0 --time_frac 1.0