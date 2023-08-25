
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
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e12/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e12.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e13/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e13.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e14/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e14.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e15/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e15.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e16/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e16.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_1_1e17/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_1_1e17.png --n_hidden 128
