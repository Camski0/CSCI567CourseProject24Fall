#%%
task_idx = 2 # TODO CHOOSE.

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle as pkl
import logging
import os
from GCCA.gcca import GCCA
import h5py

###############################################################################
# CCA
###############################################################################
class CCA(GCCA):
    def __init__(self, n_components=2, reg_param=0.1):
        GCCA.__init__(self, n_components, reg_param)
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')
        self.z_p = np.array([])

    def fit(self, x0, x1):
        x_list = [x0, x1]
        data_num = len(x_list)
        self.logger.info("data num is %d", data_num)
        for i, x in enumerate(x_list):
            self.logger.info("data shape x_%d: %s", i, x.shape)

        self.logger.info("normalizing")
        x_norm_list = [self.normalize(x) for x in x_list]

        d_list = [0] + [sum([len(x.T) for x in x_list][:i + 1]) for i in range(data_num)]
        cov_mat = self.calc_cov_mat(x_norm_list)
        cov_mat = self.add_regularization_term(cov_mat)
        c_00 = cov_mat[0][0]
        c_01 = cov_mat[0][1]
        c_11 = cov_mat[1][1]

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        left_1 = np.dot(c_01, np.linalg.solve(c_11,c_01.T))
        right_1 = c_00
        eigvals_1, eigvecs_1 = self.solve_eigprob(left_1, right_1)
        eigvecs_1_norm = self.eigvec_normalization(eigvecs_1, right_1)

        right_2 = c_11
        eigvecs_2 = 1 / eigvals_1 * np.dot(np.linalg.solve(c_11, c_01.T), eigvecs_1_norm)
        eigvecs_2_norm = self.eigvec_normalization(eigvecs_2, right_2)

        self.data_num = data_num
        self.cov_mat = cov_mat
        self.h_list = [eigvecs_1_norm, eigvecs_2_norm]
        self.eigvals = eigvals_1

    def ptransform(self, x0, x1, beta=0.5):
        x0_projected, x1_projected = self.transform(x0, x1)
        I = np.eye(len(self.eigvals))
        lamb = np.diag(self.eigvals)
        mat1 = np.linalg.solve(I - np.diag(self.eigvals**2), I)
        mat2 = -np.dot(mat1, lamb)
        mat12 = np.vstack((mat1, mat2))
        mat21 = np.vstack((mat2, mat1))
        mat = np.hstack((mat12, mat21))
        p = np.vstack((lamb**beta, lamb**(1-beta)))
        q = np.vstack((x0_projected.T, x1_projected.T))
        z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]

        self.z_p = z
        return x0_projected, x1_projected, z

    def save_params(self, filepath):
        GCCA.save_params(self, filepath)
        if len(self.z_p) != 0:
            with h5py.File(filepath, 'a') as f:
                f.create_dataset("z_p", data=self.z_p)
                f.flush()

    def load_params(self, filepath):
        GCCA.load_params(self, filepath)
        with h5py.File(filepath, "r") as f:
            if "z_p" in f:
                self.z_p = f["z_p"].value
            f.flush()

###############################################################################
# dataset and model
###############################################################################
class PathwayDataset(Dataset):
    def __init__(self, X, y, pw_dic, pw_order_dict, max_length):
        self.samples = X.index.tolist()
        self.X = X
        self.y = y.astype(int)
        self.pw_dic = pw_dic
        self.pw_order_dict = pw_order_dict
        self.max_length = max_length
        self.gene_to_col = {gene: i for i, gene in enumerate(X.columns)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        x_row = self.X.loc[sample_id].values.astype(np.float32)
        label = self.y.loc[sample_id]

        pathway_data = []
        for pw in self.pw_order_dict:
            ordered_genes = self.pw_order_dict[pw]
            indices = [self.gene_to_col[g] for g in ordered_genes if g in self.gene_to_col]
            pw_values = x_row[indices]
            if len(pw_values) < self.max_length:
                pad_length = self.max_length - len(pw_values)
                pw_values = np.concatenate([pw_values, np.zeros(pad_length, dtype=np.float32)])
            elif len(pw_values) > self.max_length:
                pw_values = pw_values[:self.max_length]
            pw_tensor = torch.tensor(pw_values)
            pathway_data.append(pw_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return pathway_data, label_tensor


from transformer import TransformerEncoder

class MultiPathwayTransformerModel(nn.Module):
    def __init__(
        self,
        max_pathway_length,
        num_layers=3,
        dropout_rate=0.15,
        classif_hidden_units=256,
        num_classes=2,
        d_v=512):

        super(MultiPathwayTransformerModel, self).__init__()
        
        d_e = max_pathway_length
        self.pw_transformer = TransformerEncoder(
            N=num_layers,
            H=2,
            D_e=d_e,
            D_qk=None,
            D_v=d_v,
            dropout=dropout_rate)

        self.classif_layer = nn.Sequential(
            nn.Linear(d_e, classif_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classif_hidden_units, num_classes)
        )

    def forward(self, pathway_data_list):
        r"""
        Args:
            pathway_data_list: List of tensors, each tensor is of shape [batch_size, max_pathway_length]
        Returns:
            X_context: [num_pathways, d_v]

        """

        X_e = torch.stack(pathway_data_list, dim=1)
        X_context = self.pw_transformer(X_e, X_e) # X_e: [batch_size, num_pathways, max_pathway_length] -> X_context: [batch_size, num_pathways, d_v]

        features = []
        X_context_temp = X_context.transpose(0, 1) # [num_pathways, batch_size, d_e]
        for i, out in enumerate(X_context_temp):
            features.append(out)

        X_context = torch.mean(X_context, dim=1) # [batch_size, d_e]
    
        logits = self.classif_layer(X_context)
        return logits, features

#%%
###############################################################################
# precompute CCA directions
###############################################################################
def precompute_cca(model, loader, device, cca, pathway_indices=(0, 1), num_batches=50):
    model.eval()
    f1_all = []
    f2_all = []
    batch_count = 0
    with torch.no_grad():
        for pw_data, labels in loader:
            pw_data = [d.to(device) for d in pw_data]
            _, features_list = model(pw_data)
            f1 = features_list[pathway_indices[0]].cpu().numpy()
            f2 = features_list[pathway_indices[1]].cpu().numpy()
            f1_all.append(f1)
            f2_all.append(f2)
            batch_count += 1
            if batch_count >= num_batches:
                break

    x0 = np.concatenate(f1_all, axis=0)
    x1 = np.concatenate(f2_all, axis=0)
    cca.fit(x0, x1)
    return cca

###############################################################################
# compute CCA penalty from canonical correlation
###############################################################################
def cca_penalty(cca, features_list, pathway_indices=(0, 1), lambda_corr=0.1):
    f1 = features_list[pathway_indices[0]].detach().cpu().numpy()
    f2 = features_list[pathway_indices[1]].detach().cpu().numpy()

    # # transform according to CCA directions
    # x0_projected, x1_projected = cca.transform(f1, f2)

    # canonical correlation is represented by cca.eigvals
    # take the mean of eigvals as overall correlation measure
    can_corr = np.mean(cca.eigvals)
    # to maximize correlation.    
    penalty_val = (1.0 - can_corr)*lambda_corr
    return torch.tensor(penalty_val, dtype=torch.float32, device=features_list[0].device)

#%%
###############################################################################
# training
###############################################################################
device = "cuda:2"  # choose accordingly

with open('./data/pw_dic.pkl', 'rb') as f:
    pw_dic = pkl.load(f)

with open(f'./data/pw_ordered_dic{task_idx}.pkl', 'rb') as f:
    pw_order_dict = pkl.load(f)

all_pathway_lengths = [len(pw_order_dict[pw]) for pw in pw_dic]
max_pathway_length = max(all_pathway_lengths)

X_trn = pd.read_csv(f'./data/X{task_idx}_trn.csv', index_col=0)
y_trn = pd.read_csv(f'./data/y{task_idx}_trn.csv', index_col=0)
X_tst = pd.read_csv(f'./data/X{task_idx}_tst.csv', index_col=0)
y_tst = pd.read_csv(f'./data/y{task_idx}_tst.csv', index_col=0)

y_trn.iloc[:,0] = y_trn.iloc[:,0].factorize()[0]
y_tst.iloc[:,0] = y_tst.iloc[:,0].factorize()[0]

# # judge if pw_order_dict.keys() is sorted. conclusion is yes.
# assert all(np.array(list(pw_order_dict.keys())) == np.unique(np.array(list(pw_order_dict.keys()))))
train_dataset = PathwayDataset(X_trn, y_trn, pw_dic, pw_order_dict, max_pathway_length)
test_dataset = PathwayDataset(X_tst, y_tst, pw_dic, pw_order_dict, max_pathway_length)

batch_size = 5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MultiPathwayTransformerModel(
    max_pathway_length=max_pathway_length,
    num_layers=3, # NOTE
    dropout_rate=0.15,
    classif_hidden_units=400, # NOTE
    num_classes=2,
    d_v=512 # NOTE
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # NOTE

#%%
cca = CCA(n_components=2, reg_param=0.1)
pathway_indices = (0, 1) 
cca = precompute_cca(model, train_loader, device, cca, pathway_indices=pathway_indices)

lambda_corr = 0.1
num_epochs = 400

#%%

best_sd = model.state_dict()
best_auc = 0.0
best_val_loss = float('inf')
best_epoch = 0
early_stopping = 50 # TODO

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    train_loss = 0.0
    total_steps = 0
    for pw_data, labels in train_loader:
        pw_data = [d.to(device) for d in pw_data]
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, features_list = model(pw_data)
        ce_loss = criterion(logits, labels.flatten())

        # add CCA penalty
        corr_pen = cca_penalty(cca, features_list, pathway_indices=pathway_indices, lambda_corr=lambda_corr)
        total_loss = ce_loss + corr_pen
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        total_steps += 1
    avg_train_loss = train_loss / total_steps

    # validation
    model.eval()
    val_loss = 0.0
    val_steps = 0
    preds_all = []
    probs_all = []
    with torch.no_grad():
        for pw_data_val, labels_val in test_loader:
            pw_data_val = [d.to(device) for d in pw_data_val]
            labels_val = labels_val.to(device)
            logits_val, _ = model(pw_data_val)
            val_ce = criterion(logits_val, labels_val.flatten())
            val_loss += val_ce.item()

            preds = torch.argmax(logits_val, dim=1)
            probs = torch.softmax(logits_val, dim=1)
            preds_all.append(preds)
            probs_all.append(probs[:, 1])
            val_steps += 1
    val_loss /= val_steps
    val_acc = (torch.cat(preds_all).cpu().numpy() == y_tst.values.flatten()).mean()
    val_auc = roc_auc_score(y_tst.values.astype(np.int32).flatten(), torch.cat(probs_all).cpu().numpy())
    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    # if val_auc > best_auc:
    if val_loss < best_val_loss:
        best_sd = model.state_dict()
        best_auc = val_auc
        best_val_loss = val_loss
        best_epoch = epoch
    else:
        early_stopping -= 1
        if early_stopping == 0:
            torch.save(best_sd, f'./ckpt/model_improved_{task_idx}.pt')
            break

print(f"Best {best_auc} at epoch {best_epoch}")

# %% 500 bootstrapping, 95CI
# task_idx = 1

X_tst = pd.read_csv(f'./data/X{task_idx}_tst.csv', index_col=0)
y_tst = pd.read_csv(f'./data/y{task_idx}_tst.csv', index_col=0)
y_tst.iloc[:,0] = y_tst.iloc[:,0].factorize()[0]
test_dataset = PathwayDataset(X_tst, y_tst, pw_dic, pw_order_dict, max_pathway_length)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_sd = torch.load(f'./ckpt/model_improved_{task_idx}.pt')
model = MultiPathwayTransformerModel(
    max_pathway_length=max_pathway_length,
    num_layers=3,
    dropout_rate=0.15,
    classif_hidden_units=400,
    num_classes=2,
    d_v=512
).to(device)
model.load_state_dict(best_sd)
model.eval()

# predictions on full test set
all_probs = []
all_labels = []
with torch.no_grad():
    for pw_data_val, labels_val in test_loader:
        pw_data_val = [d.to(device) for d in pw_data_val]
        labels_val = labels_val.to(device)
        logits_val, _ = model(pw_data_val)
        probs = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels_val.cpu().numpy())

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

# bootstrap
n_bootstraps = 500
rng = np.random.RandomState(0)
boot_scores = []

for i in range(n_bootstraps):
    # sample with replacement
    indices = rng.randint(0, len(all_labels), len(all_labels))
    if len(np.unique(all_labels[indices])) < 2:
        continue
    score = roc_auc_score(all_labels[indices], all_probs[indices])
    boot_scores.append(score)

boot_scores = np.array(boot_scores)
mean_auc = np.mean(boot_scores)
sorted_scores = np.sort(boot_scores)
lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]

print(f"Mean AUC: {mean_auc:.4f}, 95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]")

boot_scores_df = pd.DataFrame(boot_scores, columns=['AUC'])
boot_scores_df.to_csv(f'./result/model_improved_bootstrapping_{task_idx}.csv')


# %%
