
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

python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/slabsman_1c/DON_Dense_100_128_128_go0.pt --out_name slabsman_1c_go0.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/slabsman_1c/DON_Dense_100_128_128_go1.pt --out_name slabsman_1c_go1.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/slabsman_1c/DON_Dense_100_128_128_go2.pt --out_name slabsman_1c_go2.png --n_hidden 128
