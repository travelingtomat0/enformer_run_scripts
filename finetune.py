# Trick: remove versionings and remove AdamW in ./mtl/imports.py

import scanpy as sc
from geneformer import Classifier
import numpy as np
import pybiomart
import argparse
import pandas as pd

def get_ensembl_mappings():                                   
    # Set up connection to server                                               
    dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',
                    host='http://www.ensembl.org')
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default='Pancreas', help='dataset name.')
args = parser.parse_args()

adata = sc.read_h5ad(f"/home/tomap1/scratch/Olga_Data/{args.fname}.h5ad")
dataset = get_ensembl_mappings()

attributes = ['ensembl_transcript_id', 'mgi_symbol', 
                    'ensembl_gene_id', 'ensembl_peptide_id']

response = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])

response.index = list(response["Gene name"])
df_unique_index = response[~response.index.duplicated(keep='first')]

rm_genes = adata[:,~adata.var_names.isin(response["Gene name"])].var_names
ad = adata[:,adata.var_names.isin(response["Gene name"])]
ad.var["ensembl_id"] = df_unique_index.loc[list(ad.var_names)]["Gene stable ID"]
#list(response.loc[list(ad.var_names)]["Gene stable ID"])

ad.obs["n_counts"] = ad.layers["counts"].sum(axis=1)

if not args.fname == "ImmuneAtlas":
    ad.write_h5ad(f"/home/tomap1/scratch/Olga_Data/{args.fname}.h5ad")
else:
    print("Skip storage!")
from geneformer import TranscriptomeTokenizer


tk = TranscriptomeTokenizer({"CellType": "CellType", "batchlb": "individual"}, nproc=14,
                            model_input_size=2048)

if args.fname == "ImmuneAtlas":
    holdouts = ["10x 5' v2"]
else:
    holdouts = list(adata.obs.batchlb.unique())
    holdouts = holdouts[2:]

for holdout in holdouts:
    print("Running for holdout of", holdout)
    cc = Classifier(classifier="cell",
                cell_state_dict={"state_key": "CellType", "states": "all"},
                filter_data=None, # Fine tune on all data types. Example was {"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]},
                nproc=14,
                num_crossval_splits=1,
                training_args = {"num_train_epochs": 10,},# "per_device_train_batch_size": 32},
                #forward_batch_size=12,
    )
    
    train_ids=list(ad.obs.batchlb.unique())
    train_ids.remove(holdout)
    val_ids=[train_ids[-1]]
    eval_ids=[holdout]
    train_test_id_split_dict = {"attr_key": "individual",
                                "train": train_ids,
                                "eval": val_ids,
                                "test": eval_ids}
    
    tk.tokenize_data(f"/home/tomap1/scratch/{args.fname}",
                 f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer",
                 "test-1",
                 file_format="h5ad",)
    
    cc.prepare_data(input_data_file=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1.dataset",
                output_directory=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer",
                output_prefix="test-1",
                split_id_dict=train_test_id_split_dict)

    """cc.prepare_data(input_data_file=f"/home/tomap1/scratch/{args.fname}-geneformer/test-1.dataset",
                    output_directory=f"/home/tomap1/scratch/scAugment/{args.fname}-{holdout}-geneformer",
                    output_prefix="test-1",
                    split_id_dict=train_test_id_split_dict)"""
    
    all_metrics = cc.validate(model_directory="/home/tomap1/scratch/scAugment/Geneformer/gf-6L-30M-i2048/",
                          prepared_input_data_file=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1_labeled_train.dataset",
                          id_class_dict_file=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1_id_class_dict.pkl",
                          output_directory=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer",
                          output_prefix="test-1",
                          split_id_dict=train_test_id_split_dict,
                          )

    print("EVALUATE.")
    all_metrics = cc.evaluate_saved_model(
        model_directory=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/250331_geneformer_cellClassifier_test-1/ksplit1",
        id_class_dict_file=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1_id_class_dict.pkl",
        test_data_file=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1_labeled_test.dataset",
        output_directory=f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer",
        output_prefix="test-1",
    )
    
    pd.DataFrame([all_metrics["macro_f1"], all_metrics["acc"]], index=["macro_f1", "accuracy"]).to_csv(f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/submission.csv")

    from geneformer import EmbExtractor

    embex = EmbExtractor(model_type="CellClassifier",
                        num_classes=len(adata.obs.CellType.unique()),
                        #filter_data={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]},
                        #max_ncells=1000,
                        max_ncells=len(ad),
                        emb_layer=-1,
                        emb_label=["individual","CellType"],
                        labels_to_plot=["CellType"],
                        forward_batch_size=200,
                        nproc=16,
                        emb_mode="cell",
                        token_dictionary_file="/home/tomap1/scratch/scAugment/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl") # change from current default dictionary for 30M model series

    # extracts embedding from input data
    # input data is tokenized rank value encodings generated by Geneformer tokenizer (see tokenizing_scRNAseq_data.ipynb)
    # example dataset for 30M model series: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
    try:
        embs = embex.extract_embs(f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/250331_geneformer_cellClassifier_test-1/ksplit1", # example 30M fine-tuned model
                            f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/test-1.dataset",
                            f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer",
                            "test-1")
        embs.to_csv(f"/home/tomap1/scratch/{args.fname}-{holdout}-geneformer/embedding.csv")
    except:
        print("Issue building embedding.")