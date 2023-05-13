from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
import hashlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True


class ZincDataset(InMemoryDataset):
    raw_url = 'https://docs.google.com/uc?export=download&id=17kGpZOVwIGb_H57f4SvkPagdwqA8tADD'

    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['zinc_train.txt', 'zinc_dev.txt', 'zinc_test.txt']

    @property
    def split_file_name(self):
        return ['zinc_train.txt', 'zinc_dev.txt', 'zinc_test.txt']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['proc_tr_10000.pt', 'proc_val_10000.pt', 'proc_test_10000.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
        except ImportError:
            file_path = download_url(self.processed_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

    def process(self):

        RDLogger.DisableLog('rdApp.*')
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        smile_list = open(self.split_paths[self.file_idx]).readlines()[1:10000]

        data_list = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in types:
                    type_idx.append(types['C'])
                else:
                    type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            y = torch.zeros(size=(1, 0), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class ZincDataModule(MolecularDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.remove_h = True
        self.datadir = cfg.dataset.datadir
        self.train_smiles = []
        self.prepare_data()

    def prepare_data(self) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': ZincDataset(stage='train', root=root_path),
                    'val': ZincDataset(stage='val', root=root_path),
                    'test': ZincDataset(stage='test', root=root_path)}
        super().prepare_data(datasets)


class Zincinfos(AbstractDatasetInfos):
    atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5,
                    'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
    atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.name = 'Guacamol'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        self.num_atom_types = 12
        self.max_weight = 1000

        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]

        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 78.97, 12: 28.09}

        self.node_types = torch.tensor([0.7163, 0.1169, 0.1195, 0.0135, 0.0000, 0.0023, 0.0085, 0.0000, 0.0000,
                                        0.0230, 0.0000, 0.0000])

        self.edge_types = torch.tensor([8.9273e-01, 4.1264e-02, 7.3240e-03, 3.4301e-04, 5.8342e-02])

        self.n_nodes = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                     0.0000e+00, 0.0000e+00, 0.0000e+00, 1.3335e-04, 2.6669e-04, 2.0002e-04,
                                     3.0003e-04, 9.0009e-04, 3.6004e-03, 5.1005e-03, 1.3535e-02, 3.5670e-02,
                                     7.6974e-02, 1.2925e-01, 1.3858e-01, 1.4435e-01, 1.4421e-01, 1.3411e-01,
                                     1.1018e-01, 5.1605e-02, 1.1001e-02, 3.3337e-05])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0:7] = torch.tensor([0.0000, 0.1019, 0.2366, 0.3705, 0.2756, 0.0093, 0.0062])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        if recompute_statistics:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()  # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(300)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_path = os.path.join(base_path, cfg.dataset.datadir)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])
        # all_molecules = all_molecules[:10]
        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles
