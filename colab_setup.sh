#!/bin/bash

# Hardcoded path
# This could be reduced to one line but I know this works
%cd ~
%cd ..
%cd content/gdrive/MyDrive/'Personal - 3rd year'/CERI/'Christian, Tim and Timothy '/ColabDirs/SSD2

# Assuming that we are using Python 3.7
# We will first create our virtual environment if it has not been made already
sudo apt-get install python3.7-venv
python -m venv colab_venv_uno_test