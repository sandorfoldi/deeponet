
#!/bin/sh
#BSUB -J f
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
python src/train.py --dataset /work3/s216416/deeponet/data/1f --outputfolder 1f --run_name 1f --epochs 400