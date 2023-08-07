
#!/bin/sh
#BSUB -J pred
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

python src/pred.py --data /work3/s216416/deeponet/data/1a/900.npy --model_path models/1a/DON_Dense_100_128_128.pt --out_name 1a.png
python src/pred.py --data /work3/s216416/deeponet/data/1b/900.npy --model_path models/1b/DON_Dense_100_128_128.pt --out_name 1b.png
python src/pred.py --data /work3/s216416/deeponet/data/1c/900.npy --model_path models/1c/DON_Dense_100_128_128.pt --out_name 1c.png
python src/pred.py --data /work3/s216416/deeponet/data/1d/900.npy --model_path models/1d/DON_Dense_100_128_128.pt --out_name 1d.png