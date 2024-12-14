import pandas as pd
import numpy as np
from typing import Union

SPLITTER = '@'
# TCGA
DATA_PATH = "/data/zhaohong/TCGA_data/data/"
P2G = pd.read_csv(DATA_PATH  + "processed/TCGA_protein2gene_mapping.csv", index_col=0)
C2G = pd.read_csv(DATA_PATH  + "processed/TCGA_cpg2gene_mapping.csv", index_col=0)

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

def factorize_label(y):
    r"""
    Factorize the label.
    """
    y_uni = np.unique(y)
    value_to_index = {value: idx for idx, value in enumerate(y_uni)}
    y_fac = np.array([value_to_index[item] for item in y]).astype(np.int64)
    return y_fac, value_to_index

def mod_mol_dict(mol_ids):
    r"""
    
    Args:
        mol_ids (np.array): an array of molecule ids. e.g., array(['DNAm@cg22832044', 'DNAm@cg19580810', 'DNAm@cg14217534',
            'mRNA@CDH1', 'mRNA@RAB25', 'mRNA@TUBA1B', 'protein@E.Cadherin',
            'protein@Rab.25', 'protein@Acetyl.a.Tubulin.Lys40'], dtype=object)
    Returns:
        dict: a dictionary of mods, mol names, and mods uni.
    """
    mods = np.array([mod_id.split(SPLITTER)[0] for mod_id in mol_ids])
    mods_uni = np.unique(mods) # auto sorting in ascending order
    mols = np.array([mod_id.split(SPLITTER)[1] for mod_id in mol_ids])
    return {'mods': mods, 'mols': mols, 'mods_uni': mods_uni}

def explode_df_col(
    df,
    col_name
):
    df[col_name] = df[col_name].str.split(r'[;,/.]') # NOTE TODO delete the .?
    df = df.explode(col_name).reset_index(drop=True)
    return df

def count_parameters(model):
    """
    Counts the total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        total_params (int): Total number of parameters.
        trainable_params (int): Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model.__class__.__name__}: {total_params:,} total parameters, {trainable_params:,} trainable.")


def load_TCGA(
        data_path=DATA_PATH, # NOTE
        mods=None,
        load_only_rowcol_names=False,
):
    r"""

    Args:
        mods: list of str, e.g. ['CNV', 'DNAm', 'SNV', 'mRNA', 'miRNA', 'protein']

    """
    assert mods is not None
    mods = np.unique(mods) # sorted
    if not load_only_rowcol_names:
        data = [pd.read_csv(DATA_PATH + f"processed/{mod}_mat.csv", index_col=0) for mod in mods]
        return data
    else:
        cols_all = []
        rows_all = []
        for mod in mods:
            cols = pd.read_csv(DATA_PATH + f"processed/{mod}_mat.csv", index_col=0, nrows=0).columns.values
            rows = pd.read_csv(DATA_PATH + f"processed/{mod}_mat.csv", index_col=0, usecols=[0]).index.values
            cols_all.append(cols)
            rows_all.append(rows)
        return rows_all, cols_all

# TODO... not inplace
def list_pd_loc(dfs, loc):
    for i, df in enumerate(dfs):
        dfs[i] = df.loc[loc]
    return dfs

def mol_map2gene_TCGA(
    mol_ids,
):
    r"""
    
    TODO solve the one-to-many problem, which causes differences in denominators in precision (not exactly top k).

    P2G and C2G were preprocessed to only include proteins/cpgs that can be mapped to genes in preprocessed TCGA mRNA data.

    Note that there can be CpG sites mappable to multiple genes, so do proteins.

    Args:
        mol_ids: a 1d np array with mod@mol 
    Returns:
        np.array: a 1d array of mapped and intersected gene names
    """
    
    mappings = {
        'protein' : P2G,
        'DNAm' : C2G,
    }
    gene_set = np.array([])
    mmdic = mod_mol_dict(mol_ids)
    for mod in mmdic['mods_uni']:
        if mod == 'mRNA':
            gene_set = np.append(gene_set, mmdic['mols'][mmdic['mods']=='mRNA'])
            continue
        single_mod_mols = mmdic['mols'][mmdic['mods']==mod]
        # get uni genes
        uni_genes = mappings[mod].loc[single_mod_mols, 'gene'].unique()
        gene_set = np.append(gene_set, np.unique(uni_genes))
    return np.unique(gene_set)
