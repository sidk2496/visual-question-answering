# visual-question-answering
This is the final project of the course 10-707: Topics in Deep Learning offered by Carnegie Mellon University in Spring 2019.

# Dependencies:
1. keras with tensorflow-gpu backend
2. sklearn
3. numpy

# System requirements
1. Tested on Tesla K-80 GPU with 11441 MB memory.

# Instructions for running:
1. cd visual-question-answering

2. Run train.sh to train our best model:- Question attention without question training. The model resumes training from where we left off. 'best_custom_acc' is the metric to focus on while the model trains. 'best_ans_loss' should be ignored as it is just a dummy loss always returning 0.

3. Run predict.sh to make predictions on the test set using our best model:- Question attention without question training. The output file prediction.json will be saved in the src folder.

NOTE:- The scripts will download our preprocessed data files which may take a few minutes. The training and testing scripts will download some redundant files. When the prompt about whether these files should be replaced comes up, hit 'y'.
