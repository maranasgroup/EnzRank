import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import keras
from keras import backend as K
from keras.models import load_model
import argparse
import h5py
import pdb

from rdkit import Chem
from rdkit.Chem import AllChem

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0] 
    else:
        return [seq_dic[aa] for aa in seq]
    
def prot_feature_gen_from_str_input(prot_input_str, prot_len = 2500):
    Prot_ID = prot_input_str.split(':')[0]
    Prot_seq = prot_input_str.split(':')[1]
    prot_dataframe = pd.DataFrame({'Protein_ID': Prot_ID, 'Sequence': Prot_seq}, index = [0])
    prot_dataframe.set_index('Protein_ID')
    
    
    prot_dataframe["encoded_sequence"] = prot_dataframe.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    prot_feature = sequence.pad_sequences(prot_dataframe["encoded_sequence"].values, prot_len)
    
    return prot_feature, Prot_ID
    
    
KEGG_compound_read = pd.read_csv('./../CNN_data/Final_test/kegg_compound.csv', index_col = 'Compound_ID')
kegg_df = KEGG_compound_read.reset_index()
    
def mol_feature_gen_from_str_input(mol_str, kegg_id_flag):
	
	if kegg_id_flag == 1:
		KEGG_ID = mol_str
		kegg_id_loc = kegg_df.index[kegg_df.Compound_ID == KEGG_ID][0]
		KEGG_ID_info = kegg_df.loc[kegg_id_loc]
		KEGG_ID_info_df = KEGG_ID_info.to_frame().T.set_index('Compound_ID')
		
		final_return = KEGG_ID_info_df
		final_id = KEGG_ID
		
	else: 
		try:
			mol_ID = mol_str.split(':')[0]
			mol_smiles = mol_str.split(':')[1]
			mol = Chem.MolFromSmiles(mol_smiles)
			fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)
			fp_list = list(np.array(fp1).astype(float))
			fp_str = list(map(str, fp_list))
			mol_fp = '\t'.join(fp_str) 
			
			mol_dict = {}
			mol_dict['Compound_ID'] = mol_ID
			mol_dict['Smiles'] = mol_smiles
			mol_dict['morgan_fp_r2'] = mol_fp
			
			mol_info_df = pd.DataFrame(mol_dict, index=[0])
			mol_info_df.set_index('Compound_ID')
			
			final_return = mol_info_df
			final_id = mol_ID
			
		except Exception as error:
			print('Something wrong with molecule input string...' + repr(error)) 
			
	return final_return, final_id
    
    
def act_df_gen_mol_feature(mol_id, prot_id):
	act_df = pd.DataFrame({'Protein_ID':prot_id, 'Compound_ID': mol_id}, index = [0])
	
	return act_df
	
	
def compound_feature_gen_df_input(act_df, comp_df, comp_len = 2048, comp_vec= 'morgan_fp_r2'):
	act_df = pd.merge(act_df, comp_df, left_on='Compound_ID', right_index = True)
	comp_feature = np.stack(act_df[comp_vec].map(lambda fp: fp.split("\t")))
	return comp_feature


def protein_feature_gen(prot_csv_file, prot_len = 2500):
    protein_df = pd.read_csv(prot_csv_file, index_col='Protein_ID')
    protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    pfeature = sequence.pad_sequences(protein_df["encoded_sequence"].values, prot_len)
    
    return pfeature


def compound_feature_gen(act_csv_file, comp_csv_file, comp_len = 2048, comp_vec = 'morgan_fp_r2'):
    act_df = pd.read_csv(act_csv_file)
    comp_df = pd.read_csv(comp_csv_file, index_col = 'Compound_ID')
    act_df = pd.merge(act_df, comp_df, left_on='Compound_ID', right_index = True)
    comp_feature = np.stack(act_df[comp_vec].map(lambda fp: fp.split("\t")))
    return comp_feature


def load_modelfile(model_string):
	loaded_model = tf.keras.models.load_model(model_string)
	return loaded_model
	
loaded_model = load_modelfile('./../CNN_results/model_final.model')

def model_prediction(compound_feature, enz_feature):
    prediction_vals = loaded_model.predict([compound_feature, enz_feature])
    
    return prediction_vals[0][0]
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    # test_params
#     parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
#     parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
#     parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
#     parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
#     parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true", default=False)
#     parser.add_argument("--output", "-o", help="Prediction output", type=str)
#     parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
#     parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
#     parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp")
#     parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    args = parser.parse_args()
    
    model = args.model

#     test_names = args.test_name
#     tests = args.test_dti_dir
#     test_proteins = args.test_protein_dir
#     test_drugs = args.test_drug_dir
#     test_sets = zip(test_names, tests, test_drugs, test_proteins)
#     with_label = args.with_label
#     output_file = args.output


    f = h5py.File(model, 'r+')

    try:
        f.__delitem__("optimizer_weights")
    except:
        print("optimizer_weights are already deleted")

    f.close()

#     type_params = {
#         "prot_vec": args.prot_vec,
#         "prot_len": args.prot_len,
#         "drug_vec": args.drug_vec,
#         "drug_len": args.drug_len,
#     }
#     test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=with_label, **type_params)
#                 for test_name, test_dti, test_drug, test_protein in test_sets}

    loaded_model = load_model(model)
    print('model has been loaded')
#     print("prediction")
#     result_df = pd.DataFrame()
#     result_columns = []
#     for dataset in test_dic:
#         N=1
#         temp_df = pd.DataFrame()
#         prediction_dic = test_dic[dataset]
# #         pdb.set_trace()
# #         N = int(prediction_dic["drug_feature"].shape[0]/50)
# #         pdb.set_trace()
#         d_splitted = np.array_split(prediction_dic["drug_feature"], N)
#         p_splitted = np.array_split(prediction_dic["protein_feature"], N)
#         predicted = sum([np.squeeze(loaded_model.predict([d,p])).tolist() for d,p in zip(d_splitted, p_splitted)], [])
#         temp_df[dataset, 'predicted'] = predicted
#         temp_df[dataset, 'Compound_ID'] = prediction_dic["Compound_ID"]
#         temp_df[dataset, 'Protein_ID'] = prediction_dic["Protein_ID"]
#         if with_label:
#            temp_df[dataset, 'label'] = np.squeeze(test_dic[dataset]['label'])
#         result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
#         result_columns.append((dataset, "predicted"))
#         result_columns.append((dataset, "Compound_ID"))
#         result_columns.append((dataset, "Protein_ID"))
#         if with_label:
#            result_columns.append((dataset, "label"))
#     result_df.columns = pd.MultiIndex.from_tuples(result_columns)
#     print("save to %s"%output_file)
#     result_df.to_csv(output_file, index=False)
#     '''
#     predicted = loaded_model.predict([prediction_dic["drug_feature"],prediction_dic["protein_feature"]])
#     dti_dic = prediction_dic['dti']
#     dti_dic["predicted"] = predicted
#     dti_dic.to_csv(output)
#     '''
