
#!/bin/sh
#BSUB -J 1_bs2048_128_b
#BSUB -W 4:00
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
python src/train.py --dataset /work3/s216416/deeponet/data/1b --outputfolder 1_bs2048_128_b --run_name 1_bs2048_128_b --epochs 5000 --n_hidden 128  --batch_size 2048