import copy
import time
from random import sample as SAMPLE
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, \
    average_precision_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F



# class BiModalF_Gated(nn.Module):
#     def __init__(self, protein_dim, hid_dim, atom_dim, dropout, device):
#         super().__init__()
#         self.device = device
#         self.hid_dim = hid_dim
#
#         # Protein sequence processing
#         self.protein_fc = nn.Linear(protein_dim, hid_dim)
#         self.prot_gate = nn.Sequential(
#             nn.Linear(hid_dim, hid_dim),
#             nn.Sigmoid()
#         )
#
#         # Compound atom feature processing
#         self.compound_fc = nn.Linear(atom_dim, hid_dim)
#         self.drug_gate = nn.Sequential(
#             nn.Linear(hid_dim, hid_dim),
#             nn.Sigmoid()
#         )
#
#         # Classifier
#         self.fc1 = nn.Linear(hid_dim * 2, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.out = nn.Linear(512, 2)
#
#         # Pooling
#         self.prot_maxpool = nn.AdaptiveMaxPool1d(1)
#         self.drug_maxpool = nn.AdaptiveMaxPool1d(1)
#
#         # Activation and regularization
#         self.leaky_relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, protein, compound):
#         # Feature projection
#         protein_proj = self.protein_fc(protein)
#         compound_proj = self.compound_fc(compound)
#
#         # Gated feature enhancement
#         prot_gate = self.prot_gate(protein_proj)
#         prot_enhanced = protein_proj * prot_gate
#
#         drug_gate = self.drug_gate(compound_proj)
#         drug_enhanced = compound_proj * drug_gate
#
#         # Pooling
#         prot_pool = self.prot_maxpool(prot_enhanced.permute(0, 2, 1)).squeeze(-1)
#         drug_pool = self.drug_maxpool(drug_enhanced.permute(0, 2, 1)).squeeze(-1)
#
#         # Concatenate
#         pair = torch.cat([drug_pool, prot_pool], dim=1)
#
#         # Classifier
#         x = self.dropout(pair)
#         x = self.leaky_relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.fc2(x))
#         label = self.out(x)
#
#         return label

# class BiModalF_Attn(nn.Module):
#     def __init__(self, protein_dim, hid_dim, atom_dim, dropout, device, n_heads=4):
#         super().__init__()
#         self.device = device
#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#
#         # Protein sequence processing
#         self.protein_fc = nn.Linear(protein_dim, hid_dim)
#
#         # Compound atom feature processing
#         self.compound_fc = nn.Linear(atom_dim, hid_dim)
#
#         # Self-attention layers
#         self.prot_self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
#         self.drug_self_attn = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
#
#         # Layer normalization
#         self.prot_norm = nn.LayerNorm(hid_dim)
#         self.drug_norm = nn.LayerNorm(hid_dim)
#
#         # Classifier
#         self.fc1 = nn.Linear(hid_dim * 2, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.out = nn.Linear(512, 2)
#
#         # Pooling
#         self.prot_maxpool = nn.AdaptiveMaxPool1d(1)
#         self.drug_maxpool = nn.AdaptiveMaxPool1d(1)
#
#         # Activation and regularization
#         self.leaky_relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, protein, compound):
#         # Feature projection
#         protein_proj = self.protein_fc(protein)
#         compound_proj = self.compound_fc(compound)
#
#         # Self-attention enhancement
#         prot_attn_out, _ = self.prot_self_attn(protein_proj, protein_proj, protein_proj)
#         prot_enhanced = self.prot_norm(protein_proj + prot_attn_out)
#
#         drug_attn_out, _ = self.drug_self_attn(compound_proj, compound_proj, compound_proj)
#         drug_enhanced = self.drug_norm(compound_proj + drug_attn_out)
#
#         # Pooling
#         prot_pool = self.prot_maxpool(prot_enhanced.permute(0, 2, 1)).squeeze(-1)
#         drug_pool = self.drug_maxpool(drug_enhanced.permute(0, 2, 1)).squeeze(-1)
#
#         # Concatenate
#         pair = torch.cat([drug_pool, prot_pool], dim=1)
#
#         # Classifier
#         x = self.dropout(pair)
#         x = self.leaky_relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.fc2(x))
#         label = self.out(x)
#
#         return label


class BiModalF(nn.Module):
    def __init__(self, protein_dim, hid_dim, atom_dim, dropout, device):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim

        # Protein sequence processing layer
        self.protein_fc = nn.Linear(protein_dim, hid_dim)

        # Compound atom feature processing layer
        self.compound_fc = nn.Linear(atom_dim, hid_dim)

        # Attention mechanisms
        self.protein_attention = nn.Linear(hid_dim, hid_dim)
        self.compound_attention = nn.Linear(hid_dim, hid_dim)
        self.inter_attention = nn.Linear(hid_dim, hid_dim)

        # Classifier
        self.fc1 = nn.Linear(hid_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

        # Pooling layers
        self.prot_maxpool = nn.AdaptiveMaxPool1d(1)
        self.drug_maxpool = nn.AdaptiveMaxPool1d(1)

        # Activation functions and regularization
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein, compound):
        # Feature projection
        protein_proj = self.protein_fc(protein)
        compound_proj = self.compound_fc(compound)

        # Attention mechanism
        protein_att = self.protein_attention(protein_proj)
        compound_att = self.compound_attention(compound_proj)

        # Interactive attention matrix
        c_att = compound_att.unsqueeze(2)  # [batch, compound_len, 1, hid_dim]
        p_att = protein_att.unsqueeze(1)  # [batch, 1, protein_len, hid_dim]
        att_matrix = self.inter_attention(self.relu(c_att + p_att))

        # Attention weights
        compound_weights = self.sigmoid(torch.mean(att_matrix, dim=2))
        protein_weights = self.sigmoid(torch.mean(att_matrix, dim=1))

        # Weighted features
        weighted_compound = compound_proj * 0.5 + compound_proj * compound_weights
        weighted_protein = protein_proj * 0.5 + protein_proj * protein_weights

        # Global pooling
        compound_pool = self.drug_maxpool(weighted_compound.permute(0, 2, 1)).squeeze(-1)
        protein_pool = self.prot_maxpool(weighted_protein.permute(0, 2, 1)).squeeze(-1)

        # Feature concatenation
        pair = torch.cat([compound_pool, protein_pool], dim=1)

        # Classifier
        x = self.dropout(pair)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        label = self.out(x)

        return label

# class BiModalF_CNN(nn.Module):
#     def __init__(self, protein_dim, hid_dim, atom_dim, dropout, device):
#         super().__init__()
#         self.device = device
#         self.hid_dim = hid_dim
#
#         # Protein sequence processing
#         self.protein_fc = nn.Linear(protein_dim, hid_dim)
#         self.prot_conv = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
#
#         # Compound atom feature processing
#         self.compound_fc = nn.Linear(atom_dim, hid_dim)
#         self.drug_conv = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1)
#
#         # Classifier
#         self.fc1 = nn.Linear(hid_dim * 2, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.out = nn.Linear(512, 2)
#
#         # Pooling
#         self.prot_maxpool = nn.AdaptiveMaxPool1d(1)
#         self.drug_maxpool = nn.AdaptiveMaxPool1d(1)
#
#         # Activation and regularization
#         self.leaky_relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, protein, compound):
#         # Feature projection
#         protein_proj = self.protein_fc(protein)
#         compound_proj = self.compound_fc(compound)
#
#         # Convolutional feature enhancement
#         prot_conv = self.prot_conv(protein_proj.permute(0, 2, 1))
#         prot_enhanced = F.relu(prot_conv).permute(0, 2, 1)
#
#         drug_conv = self.drug_conv(compound_proj.permute(0, 2, 1))
#         drug_enhanced = F.relu(drug_conv).permute(0, 2, 1)
#
#         # Pooling
#         prot_pool = self.prot_maxpool(prot_enhanced.permute(0, 2, 1)).squeeze(-1)
#         drug_pool = self.drug_maxpool(drug_enhanced.permute(0, 2, 1)).squeeze(-1)
#
#         # Concatenate
#         pair = torch.cat([drug_pool, prot_pool], dim=1)
#
#         # Classifier
#         x = self.dropout(pair)
#         x = self.leaky_relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.fc2(x))
#         label = self.out(x)
#
#         return label

class Predictor(nn.Module):
    def __init__(self, endecoder, device, atom_dim=32):
        super().__init__()
        self.endecoder = endecoder
        self.device = device
        self.atom_dim = atom_dim

        # Preserve original weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        nn.init.xavier_uniform_(self.weight)

    def init_weight(self):
        # Preserve original weight initialization method
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, compound, protein):
        # Preserve original shape handling logic
        compound = compound.reshape(-1, self.atom_dim)
        compound = torch.unsqueeze(compound, dim=0)  # [batch size=1, atom_num, atom_dim]

        protein = torch.unsqueeze(protein, dim=0)  # [batch size=1, protein len, protein_dim]

        # Use simplified model instead of complex structure
        out = self.endecoder(protein, compound)
        return out

    def __call__(self, compound, protein, correct_interaction, train=True, confuse=False):
        # Preserve original loss calculation method
        Loss = nn.CrossEntropyLoss()
        Loss2 = nn.BCELoss()

        if train:
            # Training mode: preserve original logic
            predicted_interaction = self.forward(compound, protein)
            loss = Loss(predicted_interaction, correct_interaction)
            return predicted_interaction, loss
        else:
            # Evaluation mode: preserve original logic
            predicted_interaction = self.forward(compound, protein)

            # Preserve original result processing method
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()

            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0, 1]

            return correct_labels, predicted_labels, predicted_scores


class Trainer:
    def __init__(self, model, lr, weight_decay, batch):
        # Preserve original initialization
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                   weight_decay=weight_decay)
        self.batch = batch

    def train(self, tranloader, device, confuse=False):
        # Preserve original training loop structure
        batch = len(tranloader)
        self.model.train()
        loss_total = 0
        i = 0

        for item in tqdm(tranloader):
            for data in item:
                compounds, protein, interaction, sample = data
                compounds = compounds.to(device)
                protein = protein.to(device)
                interaction = interaction.to(device)

                y_pred, loss = self.model(compounds, protein, interaction, confuse=confuse)
                loss.backward()

                loss_total += loss.item()

            # Preserve original optimization steps
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()


class Tester:
    def __init__(self, model):
        # Preserve original initialization
        self.model = model

    def test(self, testloader, device):
        # Preserve original testing loop structure
        self.model.eval()
        T, Y, S = [], [], []

        for item in tqdm(testloader):
            for data in item:
                compounds, protein, interaction, sample = data
                compounds = compounds.to(device)
                protein = protein.to(device)
                interaction = interaction.to(device)

                correct_labels, predicted_labels, predicted_scores = self.model(
                    compounds, protein, interaction, train=False
                )

                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)

        # Preserve original metric calculation
        AUC = roc_auc_score(T, S)

        precision, recall, _ = precision_recall_curve(T, S)
        PRAUC = auc(recall, precision)
        AUPRC = average_precision_score(T, S)
        precision_val = precision_score(T, Y)
        recall_val = recall_score(T, Y)
        acc = accuracy_score(T, Y)

        return AUC, PRAUC, AUPRC, precision_val, recall_val, acc, S

    def save_AUCs(self, AUCs, filename):
        # Preserve original saving method
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        # Preserve original saving method
        torch.save(model.state_dict(), filename)