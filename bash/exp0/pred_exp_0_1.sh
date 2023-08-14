
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

python src/pred.py --data /work3/s216416/deeponet/data/1a/0.npy --model_path models/1a/DON_Dense_100_128_128.pt --out_name 1a0.png
python src/pred.py --data /work3/s216416/deeponet/data/1a/10.npy --model_path models/1a/DON_Dense_100_128_128.pt --out_name 1a10.png
python src/pred.py --data /work3/s216416/deeponet/data/1a/100.npy --model_path models/1a/DON_Dense_100_128_128.pt --out_name 1a100.png
python src/pred.py --data /work3/s216416/deeponet/data/1a/200.npy --model_path models/1a/DON_Dense_100_128_128.pt --out_name 1a200.png