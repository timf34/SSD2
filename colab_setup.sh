#!/bin/bash

# Note that this is not yet tested!!! I am just going to move on to keep things moving - I should come back to this tho

# Hardcoded path
# This could be reduced to one line but I know this works

# Also assuming that we just clone the repo fresh each time - that way its easier to git pull I think, etc.
#%cd ~
#%cd ..
#%cd content/gdrive/MyDrive/'Personal - 3rd year'/CERI/'Christian, Tim and Timothy '/ColabDirs/SSD2

# Assuming that we are using Python 3.7
# We will first create our virtual environment if it has not been made already
sudo apt-get install python3.7-venv
python -m venv colab_venv_uno_test

# Install our packages
pip install -r wsl_requirements.txt

pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# Individually install certain things
pip install stable-baselines3==1.5.0
pip install supersuit==3.5.0
pip install tensorboard
pip install opencv-python
pip install gym==0.24.0
# Colab is cuda 11.1 so hopefully this works
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html