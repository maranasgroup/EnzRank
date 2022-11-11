# EnzRank

CNN based model for enzyme-substrate activity prediction using enzyme sequence and substrate structural information

## AICHE Annual Meet 2022
CNN models to reproduce results for AICHE annual meet 2022 talk

## Requirements
- Tensorflow 2
- rdkit
- pandas
- tqdm
- numpy 
- keras 
- scikit-learn
- streamlit

## Steps to run the tool

- create a conda environment using: conda create --prefix MLenv 
- activate the created environment using: conda activate MLenv
- install packages using requirement.txt file: pip install -r requirement.txt
- install rdkit using: pip install rdkit 
- install streamlit using: pip install streamlit

The EnzRank tool is tested on Linux-based system

Three data inputs are required- proteins sequence, substrate smiles, and enzyme-substrate pairs (Look at CNN_data folder for the required input files)

For performance prediction on different splits of data (CNN_data_split folder) run. 

`python predict_with_model.py ./CNN_results/CNN_results_1/model.model -n predict -i ./CNN_data_split/CNN_data_1/test_dataset/test_act.csv -d ./CNN_data_split/CNN_data_1/test_dataset/test_compound.csv -t ./CNN_data_split/CNN_data_1/test_dataset/test_protein.csv -v Convolution -l 2500 -V morgan_fp_r2 -L 2048 -W -o ./CNN_results/CNN_results_1/output_predictions.csv`

This will output "output_prediction.csv" file in the CNN_results folder

To launch the streamlit based graphical user interface: 

move to 'Streamlit' directory using- cd Streamlit 
launch interface using command: streamlit run main.py 


