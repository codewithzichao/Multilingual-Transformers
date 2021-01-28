#!/bin/bash

pip install transformers
pip install tensorboardX
pip install torch==1.5.0
git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
