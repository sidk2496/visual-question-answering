#!/bin/bash

mkdir data
mkdir models
cd data

# download preprocessed input data
echo "Downloading preprocessed data files..."
wget -O vqa_data.zip https://www.dropbox.com/s/4yg3hsixxx7gxjn/10707_data_v2.zip?dl=0
unzip vqa_data.zip

# download trained question attention model
echo "Downloading trained model..."
cd ../models
wget -O model.h5 https://www.dropbox.com/s/8cma2a1khlhyqrf/ques_attention_wo_ques.h5?dl=0

# run predict script
cd ../src
echo "Start testing..."
python predict.py --model_type ques_att --extracted --dest_path prediction --model_path ../models/model


