# EnzRank

CNN based model for enzyme-substrate activity prediction using enzyme sequence and substrate structural information

## ECI-conf-2022
CNN models to reproduce results for eci conference 2022 poster

## Requirements
- Tensorflow 1.15.0
- rdkit
- pandas
- tqdm
- numpy 
- keras 2.3.1
- scikit-learn

Three data inputs are required- proteins sequence, substrate smiles, and enzyme-substrate pairs (Look at CNN_data folder for the required input files)

For performance prediction on different splits of data (CNN_data folder) run. 

`python predict_with_model.py ./CNN_results/CNN_results_1/model.model -n predict -i ./CNN_data_split/CNN_data_1/test_dataset/test_act.csv -d ./CNN_data_split/CNN_data_1/test_dataset/test_compound.csv -t ./CNN_data_split/CNN_data_1/test_dataset/test_protein.csv -v Convolution -l 2500 -V morgan_fp_r2 -L 2048 -W -o ./CNN_results/CNN_results_1/output_predictions.csv`

This will output "output_prediction.csv" file in the CNN_results folder



