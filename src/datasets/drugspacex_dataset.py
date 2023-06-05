import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import PandasTools

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar

from src import utils
from src.analysis.rdkit_functions import compute_molecular_metrics
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


atom_decoder = ['H', 'C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}


class DrugSpaceXDataset(InMemoryDataset):
    raw_url = 'https://drugspacex.simm.ac.cn/static/gz/DrugSpaceX-Drug-set-smiles.smi.tar.gz'

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
        return ['DrugSpaceX-Drug-set-smiles.smi']

    @property
    def split_file_name(self):
        return ['train.smiles', 'test.smiles', 'val.smiles']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['proc_tr.pt', 'proc_val.pt', 'proc_test.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_tar(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_tar(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[0], usecols=['SMILES'], sep='\t')

        dataset = dataset.loc[dataset['SMILES'].str.len() < 130]
        dataset = dataset[
            ~dataset['SMILES'].str.contains(
                'K|Al|As|Sb|Bi|Sr|Co|V|Mg|Ca|Cr|Lu|Gd|Ga|Ra|Fe|Cu|Zn|Sm|Na|Mn|Au|Ag|In|Hg|Sn|Tc'
            )
        ]

        n_samples = len(dataset)
        n_train = int(0.8 * n_samples)
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        np.savetxt(os.path.join(self.raw_dir, self.split_file_name[0]), train.to_numpy(), fmt='%s')
        np.savetxt(os.path.join(self.raw_dir, self.split_file_name[1]), val.to_numpy(), fmt='%s')
        np.savetxt(os.path.join(self.raw_dir, self.split_file_name[2]), test.to_numpy(), fmt='%s')

    def process(self):

        RDLogger.DisableLog('rdApp.*')
        types = atom_encoder
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        smile_list = open(self.split_paths[self.file_idx]).readlines()

        data_list = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
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


class DrugSpaceXModule(MolecularDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.remove_h = True
        self.datadir = cfg.dataset.datadir
        self.train_smiles = []
        self.prepare_data()

    def prepare_data(self) -> None:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': DrugSpaceXDataset(stage='train', root=root_path),
                    'val': DrugSpaceXDataset(stage='val', root=root_path),
                    'test': DrugSpaceXDataset(stage='test', root=root_path)}
        super().prepare_data(datasets)


class DrugSpaceXInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.name = 'DrugSpaceX'
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        self.num_atom_types = 13
        self.max_weight = 1000

        self.valencies = [1, 4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]

        self.atom_weights = {1: 1, 2: 12, 3: 14, 4: 16, 5: 19, 6: 10.81, 7: 79.9,
                             8: 35.45, 9: 126.9, 10: 30.97, 11: 30.07, 12: 78.97, 13: 28.09}

        self.node_types = torch.tensor([1.5911e-04, 7.3261e-01, 9.0323e-02, 1.3962e-01, 1.0899e-02, 1.3259e-04,
                                        8.2208e-04, 8.5126e-03, 2.5458e-03, 1.5646e-03, 1.2570e-02, 2.6519e-05,
                                        2.1215e-04])

        self.edge_types = torch.tensor([9.1975e-01, 4.7786e-02, 6.0592e-03, 1.2765e-04, 2.6281e-02])

        self.n_nodes = torch.tensor([0.0000, 0.0000, 0.0005, 0.0015, 0.0060, 0.0045, 0.0055, 0.0090, 0.0110,
                                     0.0160, 0.0235, 0.0225, 0.0240, 0.0245, 0.0275, 0.0245, 0.0345, 0.0380,
                                     0.0315, 0.0450, 0.0440, 0.0509, 0.0539, 0.0430, 0.0375, 0.0380, 0.0350,
                                     0.0380, 0.0360, 0.0300, 0.0315, 0.0265, 0.0245, 0.0245, 0.0160, 0.0195,
                                     0.0145, 0.0105, 0.0060, 0.0085, 0.0105, 0.0060, 0.0080, 0.0070, 0.0030,
                                     0.0055, 0.0040, 0.0005, 0.0010, 0.0020, 0.0020, 0.0030, 0.0010, 0.0010,
                                     0.0030, 0.0010, 0.0000, 0.0000, 0.0000, 0.0005, 0.0005, 0.0005, 0.0005,
                                     0.0010, 0.0000, 0.0000, 0.0000, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000,
                                     0.0000, 0.0000, 0.0005])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0:7] = torch.tensor(
            [2.5437e-04, 1.5531e-01, 2.9161e-01, 3.1800e-01, 2.2482e-01, 6.0836e-03, 3.9215e-03]
        )

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
