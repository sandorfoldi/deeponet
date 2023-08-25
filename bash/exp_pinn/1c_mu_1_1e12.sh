
#!/bin/sh
#BSUB -J pinn_1c_mu_1_1e12
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
python src/train_pinn.py --dataset /work3/s216416/deeponet/data/1c --outputfolder pinn_1c_mu_1_1e12 --run_name pinn_1c_mu_1_1e12 --mu_boundary 1.0 --mu_colloc 1e12 --epochs 5000 --n_hidden 128  --batch_size 2048