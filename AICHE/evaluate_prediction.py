import pandas as pd
import numpy as np
import os 
import sys
from tqdm import tqdm


def evaluate_models(pred_file_path, threshold):
    pred_file = pred_file_path.split('/')[-1]
    pred_dti_df = pd.read_csv(pred_file_path, skiprows=1, usecols=[0,1,2,3])
    pdti_df = pred_dti_df.sort_values(by=["Protein_ID", "Compound_ID"]).drop_duplicates(subset = ["Protein_ID", "Compound_ID"], keep=False)
    
    pos_df = pdti_df.loc[pdti_df.label==1]
    pos_pred = sum(pos_df.predicted.values>threshold)/ len(pos_df)
    
    neg_df = pdti_df.loc[pdti_df.label==0]
    neg_pred = sum(neg_df.predicted.values<=threshold)/ len(neg_df)
    
    if len(pred_file.split('_')) == 7:
        split_name = pred_file.split('.')[0].split('_')
        epoch = split_name[3]
        dropoutProbability = split_name[-2] + '.' + split_name[-1]
        dataType = split_name[0]

    elif len(pred_file.split('_')) == 6:
        split_name = pred_file.split('.')[0].split('_')
        epoch = split_name[3]
        dropoutProbability = split_name[-1]
        dataType = split_name[0]
        
    return(pos_pred, neg_pred, epoch, dropoutProbability, dataType)
    
    
th = 0.5


csv_files = !ls ./CNN_results/Hyp_opt/epochs/model_prediction/*.csv


training_dict = {}
test_dict = {}
validation_dict = {}

test_pos = []
test_neg = []
test_nepoch = []
test_dropout = []
test_dtype = []

training_pos = []
training_neg = []
training_nepoch = []
training_dropout = []
training_dtype = []


validation_pos = []
validation_neg = []
validation_nepoch = []
validation_dropout = []
validation_dtype = []



for files in tqdm(csv_files):
    postive, negative, n_ep, dp, d_type = evaluate_models(files, th)
    
    if d_type == 'test':
        test_pos.append(postive)
        test_neg.append(negative)
        test_nepoch.append(n_ep)
        test_dropout.append(dp)
        test_dtype.append(d_type)
    elif d_type == 'training':
        training_pos.append(postive)
        training_neg.append(negative)
        training_nepoch.append(n_ep)
        training_dropout.append(dp) 
        training_dtype.append(d_type)
    elif d_type == 'validation':
        validation_pos.append(postive)
        validation_neg.append(negative)
        validation_nepoch.append(n_ep)
        validation_dropout.append(dp) 
        validation_dtype.append(d_type)
        
        
        
        
test_dict = {'Type': test_dtype, 'Positive': test_pos, 'Negative': test_neg, 'epoch': test_nepoch, 'dropout': test_dropout}


training_dict = {'Type': training_dtype, 'Positive': training_pos, 'Negative': training_neg, 'epoch': training_nepoch, 'dropout': training_dropout}

validation_dict = {'Type': validation_dtype, 'Positive': validation_pos, 'Negative': validation_neg, 'epoch': validation_nepoch, 'dropout': validation_dropout}


test_df = pd.DataFrame(test_dict)
training_df = pd.DataFrame(training_dict)
validation_df = pd.DataFrame(validation_dict)



        
        

