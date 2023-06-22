# get_started.sh

# Activate virtual environment
source env/bin/activate

# Generate data
python3 src/wave_generator.py --root data/a --mode bvp --n_ic 1000 --n_sensors 10  --d_t 0.1 --n_t 100 --x0 0 --x1 1 --d_x 0.1 
