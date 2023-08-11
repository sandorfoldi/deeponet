
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

# python src/pred.py --data /work3/s216416/deeponet/data/1e/900.npy --model_path models/1e/DON_Dense_100_128_128.pt --out_name 1e.png
# python src/pred.py --data /work3/s216416/deeponet/data/1f/900.npy --model_path models/1f/DON_Dense_100_128_128.pt --out_name 1f.png
# python src/pred.py --data /work3/s216416/deeponet/data/1g/900.npy --model_path models/1g/DON_Dense_100_128_128.pt --out_name 1g.png
# python src/pred.py --data /work3/s216416/deeponet/data/1h/900.npy --model_path models/1h/DON_Dense_100_128_128.pt --out_name 1h.png

python src/pred.py --data /work3/s216416/deeponet/data/1e/900.npy --model_path models/e64bs2048/DON_Dense_100_64_64.pt --out_name 1e64bs2048.png --n_hidden 64