#!/bin/bash

# https://zenodo.org/api/records/7065336
echo "Downloading Model checkpoints..."
wget https://zenodo.org/api/files/f8ac69c4-6f2f-4da8-a29b-870fd4dbdc84/model_ckpts.zip
mkdir finetuned_ckpts
mv model_ckpts.zip finetuned_ckpts
cd finetuned_ckpts
unzip model_ckpts.zip
cd ..

# https://zenodo.org/record/8002087
echo "Downloading LLM experiment prompt and generations..."
wget https://zenodo.org/record/8002087/files/llm-exp.zip
unzip -q llm-exp.zip

echo "Downloading Model checkpoints trained with low amount of data..."
# wget https://zenodo.org/record/8002087/files/models-ckpt-low-data.zip
mkdir model-ckpt-with-low-data
mv models-ckpt-low-data.zip model-ckpt-with-low-data
cd model-ckpt-with-low-data
unzip models-ckpt-low-data.zip
cd ..

# # To download pretrained CuBERT checkpoint, uncomment
# echo "Downloading pretrained checkpoints for training..."
# wget https://zenodo.org/record/8002087/files/pretrained_models.zip
# unzip -q pretrained_models.zip


echo "Installing requirements."
pip3 install -r requirements.txt
pip3 install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo "Preparing twostep sampled test data"
python3 get_sampled_data.py