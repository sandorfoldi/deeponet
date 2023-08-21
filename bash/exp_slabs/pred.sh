
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

python src/pred.py --data /work3/s216416/deeponet/data/1e/100.npy --model_path models/slabs_1e/DON_Dense_100_128_128.pt --out_name slabs_1_e.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1f/100.npy --model_path models/slabs_1f/DON_Dense_100_128_128.pt --out_name slabs_1_f.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1g/100.npy --model_path models/slabs_1g/DON_Dense_100_128_128.pt --out_name slabs_1_g.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1h/100.npy --model_path models/slabs_1h/DON_Dense_100_128_128.pt --out_name slabs_1_h.png --n_hidden 128