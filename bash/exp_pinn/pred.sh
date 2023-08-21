
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
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_00/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_00.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_02/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_02.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_04/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_04.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_06/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_06.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_08/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_08.png --n_hidden 128
python src/pred.py --data /work3/s216416/deeponet/data/1c/100.npy --model_path models/pinn_1c_mu_10/DON_Dense_100_128_128.pt --out_name pinn_1c_mu_10.png --n_hidden 128