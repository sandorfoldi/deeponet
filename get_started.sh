# get_started.sh

# Create a virtual environment and install dependencies
pip3 install virtualenv
python3 -m virtualenv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Generate data
python src/wave_generator.py --root data/10 --n_ic 10
python src/wave_generator.py --root data/100 --n_ic 100
python src/wave_generator.py --root data/1000 --n_ic 1000

# Visualize sample data
python src/viz.py --data data/10/0.npy --mode animate