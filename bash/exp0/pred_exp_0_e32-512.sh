
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

python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/e32bs2048/DON_Dense_100_32_32.pt --out_name 1e32bs2048.png --n_hidden 32
python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/e64bs2048/DON_Dense_100_64_64.pt --out_name 1e64bs2048.png --n_hidden 64
python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/e128bs2048/DON_Dense_100_128_128.pt --out_name 1e128bs2048.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/e256bs2048/DON_Dense_100_256_256.pt --out_name 1e256bs2048.png --n_hidden 256
python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/e512bs2048/DON_Dense_100_512_512.pt --out_name 1e512bs2048.png --n_hidden 512
