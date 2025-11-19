

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import KFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset splitting settings')
    parser.add_argument('--dataset', type=str, default="Davis",
                        choices=["drugbank", "Davis", "KIBA"],
                        help='Select dataset for training')
    parser.add_argument('--split_settings',
                        type=str, default="cold-protein",
                        choices=["random", "cold-protein"],
                        help='Select split settings: random or cold-protein (new proteins)')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    # Create main directory
    data_path = os.path.join('../data', args.dataset)
    dir_path = os.path.join(data_path, args.split_settings)
    os.makedirs(dir_path, exist_ok=True)

    # Create fold subdirectories
    for fold in range(args.folds):
        fold_path = os.path.join(dir_path, f'fold{fold}')
        os.makedirs(fold_path, exist_ok=True)
        for subdir in ["train", "valid", "test"]:
            os.makedirs(os.path.join(fold_path, subdir), exist_ok=True)

    # Load dataset
    data_file = os.path.join('../data', args.dataset, "fulldata.csv")
    full = pd.read_csv(data_file)

    # Create SMILES and Protein mapping dictionaries
    unique_smiles = full['SMILES'].unique()
    unique_proteins = full['Protein'].unique()
    smiles_to_idx = {smi: idx for idx, smi in enumerate(unique_smiles)}
    protein_to_idx = {prot: idx for idx, prot in enumerate(unique_proteins)}

    # Convert entire dataset to index representation
    full['smiles_idx'] = full['SMILES'].map(smiles_to_idx)
    full['protein_idx'] = full['Protein'].map(protein_to_idx)

    # Build sample array [smiles_idx, protein_idx, Y]
    samples = full[['smiles_idx', 'protein_idx', 'Y']].values


    if args.split_settings == 'cold-protein':
        # Cold-start split - only for new proteins
        protein_kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

        # Get unique protein indices
        unique_proteins = full['protein_idx'].unique()

        # Generate protein fold splits
        protein_folds = list(protein_kf.split(unique_proteins))

        for fold in range(args.folds):
            print(f"Processing cold-protein fold {fold}")
            fold_path = os.path.join(dir_path, f'fold{fold}')

            # Current fold's test protein indices
            _, test_protein_idx = protein_folds[fold]
            test_proteins = unique_proteins[test_protein_idx]

            # Test set: all interactions containing test proteins
            test_mask = np.isin(samples[:, 1], test_proteins)
            test_set = samples[test_mask]

            # Remaining data for training and validation
            train_val_set = samples[~test_mask]

            # Split validation set from remaining data (20%)
            np.random.shuffle(train_val_set)
            val_size = int(0.2 * len(train_val_set))
            val_set = train_val_set[:val_size]
            train_set = train_val_set[val_size:]

            # Save as DataFrame
            pd.DataFrame(train_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'train', 'samples.csv'), index=False)
            pd.DataFrame(val_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'valid', 'samples.csv'), index=False)
            pd.DataFrame(test_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'test', 'samples.csv'), index=False)

    elif args.split_settings == 'random':
        # Random split
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(samples)):
            print(f"Processing random fold {fold}")
            fold_path = os.path.join(dir_path, f'fold{fold}')

            test_set = samples[test_idx]
            train_val_set = samples[train_val_idx]

            # Split validation set from training-validation set (20%)
            np.random.shuffle(train_val_set)
            val_size = int(0.2 * len(train_val_set))
            val_set = train_val_set[:val_size]
            train_set = train_val_set[val_size:]

            pd.DataFrame(train_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'train', 'samples.csv'), index=False)
            pd.DataFrame(val_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'valid', 'samples.csv'), index=False)
            pd.DataFrame(test_set, columns=['smiles', 'sequence', 'interactions']).to_csv(
                os.path.join(fold_path, 'test', 'samples.csv'), index=False)