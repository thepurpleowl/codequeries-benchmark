#!/bin/bash

echo "Downloading Model checkpoints."
# wget https://zenodo.org/api/records/7065336
wget https://zenodo.org/api/files/f8ac69c4-6f2f-4da8-a29b-870fd4dbdc84/model_ckpts.zip

mkdir finetuned_ckpts
mv model_ckpts.zip finetuned_ckpts
cd finetuned_ckpts
unzip model_ckpts.zip

echo "Installing requirements."
pip3 install -r requirements.txt
pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113