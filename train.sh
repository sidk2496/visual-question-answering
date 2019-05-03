#!/bin/bash

mkdir data
mkdir models
mkdir train_log
cd data

# download preprocessed input data
echo "Downloading preprocessed data files..."
wget -O vqa_data.zip https://www.dropbox.com/s/4yg3hsixxx7gxjn/10707_data_v2.zip?dl=0
unzip vqa_data.zip

# download training annotations
echo "Downloading training annotations..."
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip

# download trained question attention model
echo "Downloading trained model..."
cd ../models
wget -O model.h5 https://www.dropbox.com/s/8cma2a1khlhyqrf/ques_attention_wo_ques.h5?dl=0

# run training script
cd ../src
echo "Start training..."
python train.py --model_type ques_attention --extracted --load_model ../models/model