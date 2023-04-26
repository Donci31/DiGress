from torch_geometric.data import InMemoryDataset

from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos


class DrugSpaceXDataset(InMemoryDataset):
    raw_url = 'https://drugspacex.simm.ac.cn/static/gz/DrugSpaceX-Drug-set-smiles.smi.tar.gz'
    raise NotImplemented()


class DrugSpaceXModule(MolecularDataModule):
    raise NotImplemented()


class DrugSpaceXinfos(AbstractDatasetInfos):
    raise NotImplemented()