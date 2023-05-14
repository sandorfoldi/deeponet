# get_started.sh

# Activate virtual environment
source env/bin/activate

# Generate data
python3 src/wave_generator.py --root data/s10 --n_ic 10 --n_sensors 10
python3 src/wave_generator.py --root data/s100 --n_ic 100
python3 src/wave_generator.py --root data/s1000 --n_ic 1000

python3 src/wave_generator.py --root data/f10 --n_ic 10 --n_sensors 10000 --sensor_type='fourier' --n_fourier_components=100
python3 src/wave_generator.py --root data/f100 --n_ic 100 --n_sensors 10000 --sensor_type='fourier' --n_fourier_components=100
python3 src/wave_generator.py --root data/f1000 --n_ic 1000 --n_sensors 10000 --sensor_type='fourier' --n_fourier_components=100


# Visualize sample data
python3 src/viz.py --data data/f10/0.npy --mode animate