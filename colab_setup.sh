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

# We actually don't even need to use the venv if we just git clone each time we run a Colab.
#sudo apt-get install python3.7-venv
#python -m venv colab_venv_uno_test

# Install our packages
pip install -r wsl_requirements.txt

# Colab already has 1.12 with cuda 11.3 installed. 
# pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# Individually install certain things

#pip install stable-baselines3==1.5.0
#pip install supersuit==3.5.0
pip install tensorboard
pip install opencv-python
pip install git+https://github.com/Rohan138/marl-baselines3
pip install gym==0.23.1
# For some reason its not being installed in the `wsl_requirements.txt` file
pip install wandb
pip install pygame
pip install pymunk
pip install gputil
pip install ray[rllib]==2.0.0

# Colab comes with 1.12.0 by default, although we have 1.11.0 locally!
# If Colab:
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# Else:
# pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
