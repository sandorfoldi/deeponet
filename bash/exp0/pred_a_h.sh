
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

python src/pred.py --data /work3/s216416/deeponet/data/1a/100.npy --model_path models/exp3_1a/DON_Dense_100_128_128.pt --out_name _1a.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1b/100.npy --model_path models/1b/DON_Dense_100_128_128.pt --out_name _1b.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/1c/DON_Dense_100_128_128.pt --out_name _1c.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1d/100.npy --model_path models/1d/DON_Dense_100_128_128.pt --out_name _1d.png --n_hidden 128
