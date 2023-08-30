
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

python src/pred.py --data /work3/s216416/deeponet/data/2a/100.npy --model_path models/exp2_a/DON_Dense_1000_128_128.pt --out_name exp2_a.png --n_hidden 128 --num_sensors 1000
python src/pred.py --data /work3/s216416/deeponet/data/2b/100.npy --model_path models/exp2_b/DON_Dense_1000_128_128.pt --out_name exp2_b.png --n_hidden 128 --num_sensors 1000
python src/pred.py --data /work3/s216416/deeponet/data/2c/100.npy --model_path models/exp2_c/DON_Dense_1000_128_128.pt --out_name exp2_c.png --n_hidden 128 --num_sensors 1000
python src/pred.py --data /work3/s216416/deeponet/data/2d/100.npy --model_path models/exp2_d/DON_Dense_1000_128_128.pt --out_name exp2_d.png --n_hidden 128 --num_sensors 1000
