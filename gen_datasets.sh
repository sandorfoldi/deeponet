# get_started.sh

# Activate virtual environment
source env/bin/activate

# Generate data
python3 src/wave_generator_gm.py --root data/a --mode bvp --n_ic 1 --n_sensors 1  --d_t 0.1 --n_t 100 --x0 0 --x1 3.14 --d_x 0.001 --c 1
