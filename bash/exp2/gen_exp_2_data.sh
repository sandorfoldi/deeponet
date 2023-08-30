#!/bin/sh
#BSUB -J gen
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

python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2a --m_min 10 --m_max 90 --v_min 10 --v_max 10 --num_g 1 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2b --m_min 10 --m_max 90 --v_min 10 --v_max 10 --num_g 10 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2c --m_min 20 --m_max 80 --v_min 1 --v_max 20 --num_g 10 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2d --m_min 20 --m_max 80 --v_min 1 --v_max 20 --num_g 100 --n_sensors 1000

python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2e --m_min 20 --m_max 80 --v_min 200 --v_max 200 --num_g 1 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2f --m_min 20 --m_max 80 --v_min 100 --v_max 100 --num_g 1 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2g --m_min 20 --m_max 80 --v_min 50 --v_max 50 --num_g 1 --n_sensors 1000
python src/wave_generator_gm.py --root /work3/s216416/deeponet/data/2h --m_min 20 --m_max 80 --v_min 20 --v_max 20 --num_g 1 --n_sensors 1000

