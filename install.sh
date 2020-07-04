#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh
echo "PATH=$PATH:~/miniconda/bin" >> ~/.bashrc
conda env create -f ~/IQT_tutorial/environment.yaml
source activate iqt
pip install -r ~/IQT_tutorial/requirements.txt
