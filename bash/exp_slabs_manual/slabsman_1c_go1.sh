
#!/bin/sh
#BSUB -J slabsman_1c_go1
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
python src/train_pinn.py --dataset /work3/s216416/deeponet/data/1c --outputfolder slabsman_1c_v1 --run_name slabsman_1_go1 --epochs 1000 --n_hidden 128  --batch_size 2048 --time_frac 0.4 --model_path models/slabsman_1c_v1/DON_Dense_100_128_128_go0.pt