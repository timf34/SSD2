#!/bin/bash

# Note that this is not yet tested!!! I am just going to move on to keep things moving - I should come back to this tho

# Hardcoded path
# This could be reduced to one line but I know this works
%cd ~
%cd ..
%cd content/gdrive/MyDrive/'Personal - 3rd year'/CERI/'Christian, Tim and Timothy '/ColabDirs/SSD2

# Assuming that we are using Python 3.7
# We will first create our virtual environment if it has not been made already
sudo apt-get install python3.7-venv
python -m venv colab_venv_uno_test

# Install our packages
pip install -r wsl_requirements.txt

# Individually install certain things
pip install stable-baselines3==1.5.0
pip install supersuit==3.5.0
pip install tensorboard
pip install opencv-python
pip install gym==0.24.0