
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

python src/pred.py --data /work3/s216416/deeponet/data/1a/100.npy --model_path models/exp3_1a/DON_Dense_100_128_128.pt --out_name exp3_1a.png --n_hidden 128 --num_sensors 100
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/exp3_1c/DON_Dense_100_128_128.pt --out_name exp3_1c.png --n_hidden 128 --num_sensors 100
python src/pred.py --data /work3/s216416/deeponet/data/2a/100.npy --model_path models/exp3_2a/DON_Dense_1000_128_128.pt --out_name exp3_2a.png --n_hidden 128 --num_sensors 1000
python src/pred.py --data /work3/s216416/deeponet/data/2c/100.npy --model_path models/exp3_2c/DON_Dense_1000_128_128.pt --out_name exp3_2c.png --n_hidden 128 --num_sensors 1000

