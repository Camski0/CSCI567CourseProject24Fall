#%%
###############################################################################
# import
###############################################################################
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from utils import *
import pickle as pkl

RESULT_PATH = "./result/"

#%% 
###############################################################################
# load data
###############################################################################
DATA_PATH = "/data/zhaohong/TCGA_data/data/processed/"

mods_sel = ['mRNA']
data0 = load_TCGA(mods=mods_sel, data_path=DATA_PATH)
data = data0[0].copy()

s = pd.read_csv(DATA_PATH + 'sample_info.tsv', sep='\t', index_col=0)
c = pd.read_csv(DATA_PATH + "clinical.csv", index_col=0)
c = c[['project_id', 'ajcc_pathologic_stage']]
sel_projs = ['TCGA-OV', 'TCGA-LIHC', 'TCGA-KIRC'] # KIRC's pathologic stage distribution is more suitable according to the setup in the paper

#%% convert sample level to patient level
mask = (s.loc[data.index, 'Sample Type']!='Solid Tissue Normal').values
data = data.loc[mask]
data.index = ['-'.join(x.split('-')[:-1]) for x in data.index]
data = data.groupby(data.index).mean()

#%% select gene set (present in pathways)
files = open('./data/20230205_kegg_hsa.gmt', encoding='utf-8')
files = files.readlines()
pw_dic = {}
for i in files: 
    pw_dic[i.split('\t')[0].split('_')[0]] = i.replace('\n','').split('\t')[2:] 

gset = set()
for pw in pw_dic:
    gset.update(pw_dic[pw])

#%% prepare X, y for task 1: Kidney cancer stage classification
X1 = data.loc[c.loc[data.index, 'project_id']=='TCGA-KIRC']
y1 = c.loc[X1.index, 'ajcc_pathologic_stage']
y1 = y1.replace({'Stage I': 'early', 'Stage II': 'early', 'Stage III': 'late', 'Stage IV': 'late'})
# remvoe '--' in y1
mask = y1!="'--"
y1 = y1.loc[mask]
X1 = X1.loc[mask]

X1 = X1.loc[:, X1.columns.isin(gset)]

X1.to_csv('./data/X1.csv')
y1.to_csv('./data/y1.csv')

#%% prepare X, y for task 2: liver and kidney cancer classification
X2 = data.loc[c.loc[data.index, 'project_id'].isin(['TCGA-LIHC', 'TCGA-KIRC'])]
y2 = c.loc[X2.index, 'project_id']
y2 = y2.replace({'TCGA-LIHC': 'liver', 'TCGA-KIRC': 'kidney'})

X2 = X2.loc[:, X2.columns.isin(gset)]

X2.to_csv('./data/X2.csv')
y2.to_csv('./data/y2.csv')

#%% prepare X, y for task 3: Ovarian cancer survival classification
sv = pd.read_csv(DATA_PATH + 'survival.csv', index_col=0)
sv = sv.loc[c.loc[sv.index, 'project_id']=='TCGA-OV']
sv = sv.loc[~((sv['T'] < 365*2) & (sv['E']==0))] # remove. E==0 means no event, i.e. censored
sv['y'] = '0'
sv.loc[(sv['T'] < 365*2) & (sv['E']==1), 'y'] = 'deceased'
sv.loc[(sv['T'] >= 365*5), 'y'] = 'alive'
sv = sv.loc[sv['y']!='0']
y3 = sv['y']
sids = np.intersect1d(data.index, y3.index)
y3 = y3.loc[sids]
X3 = data.loc[sids]
print(y3.value_counts())

X3 = X3.loc[:, X3.columns.isin(gset)]

X3.to_csv('./data/X3.csv')
y3.to_csv('./data/y3.csv')

# X3 = pd.read_csv('./data/X3.csv', index_col=0)
# y3 = pd.read_csv('./data/y3.csv', index_col=0)

#%% update gset

assert all(X1.columns==X2.columns) and all(X2.columns==X3.columns)

# features
g2rm = set()
gset = set(X1.columns)
for i in range(3):
    X = pd.read_csv(f'./data/X{i+1}.csv', index_col=0)
    g2rm.update(X.loc[:, (X.isna().sum(axis=0)!=0) | (X.var(axis=0)==0)].columns.values)
for cur_g2rm in g2rm:
    gset.remove(cur_g2rm)

X1 = X1.loc[:, X1.columns.isin(gset)]
X2 = X2.loc[:, X2.columns.isin(gset)]
X3 = X3.loc[:, X3.columns.isin(gset)]
X1.to_csv('./data/X1.csv')
X2.to_csv('./data/X2.csv')
X3.to_csv('./data/X3.csv')

# remove genes not in gset from pw_dic
for pw in pw_dic:
    pw_dic[pw] = [g for g in pw_dic[pw] if g in gset]
# save pw_dic
with open('./data/pw_dic.pkl', 'wb') as f:
    pkl.dump(pw_dic, f)

# save gset
with open('./data/gset.pkl', 'wb') as f:
    pkl.dump(gset, f)

#%% reorder genes in pathways accoring to the spearman corr with p-norm
for i in range(3):
    pw_ordered_dic = {}
    print(f"Processing dataset {i+1}")
    X = pd.read_csv(f'./data/X{i+1}.csv', index_col=0)
    for j, pw in enumerate(pw_dic):
        print(f"Processing pathway {j+1}/{len(pw_dic)}")
        # compute spearman correlation
        corrs = X.loc[:, pw_dic[pw]].corr(method='spearman')
        # compute p-norm of each row of corrs, where p is the number of columns (#genes)
        p = corrs.shape[1]
        p_norms = np.linalg.norm(corrs, ord=p, axis=1)
        # rank genes in decreasing order, and store in pw_ordered_dic
        pw_ordered_dic[pw] = corrs.index[np.argsort(p_norms)[::-1]]
    # save pw_ordered_dic for dataset {i+1}
    with open(f'./data/pw_ordered_dic{i+1}.pkl', 'wb') as f:
        pkl.dump(pw_ordered_dic, f)

#%% split into train and test and scale
for i in range(3):
    # if i != 0: continue
    X = pd.read_csv(f'./data/X{i+1}.csv', index_col=0)
    y = pd.read_csv(f'./data/y{i+1}.csv', index_col=0)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
    for tr_idx, tst_idx in sss.split(X, y):
        X_trn, X_tst = X.iloc[tr_idx], X.iloc[tst_idx]
        y_trn, y_tst = y.iloc[tr_idx], y.iloc[tst_idx]
        scaler = StandardScaler()
        X_trn.loc[:, :] = scaler.fit_transform(X_trn)
        X_tst.loc[:, :] = scaler.transform(X_tst)
        X_trn.to_csv(f'./data/X{i+1}_trn.csv')
        X_tst.to_csv(f'./data/X{i+1}_tst.csv')
        y_trn.to_csv(f'./data/y{i+1}_trn.csv')
        y_tst.to_csv(f'./data/y{i+1}_tst.csv')

# %%
