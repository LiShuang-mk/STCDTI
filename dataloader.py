import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import (
    smiles_to_bigraph,
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
)
from utils import integer_label_protein, smiles2onehot
import pdb


class DTIDataset(data.Dataset):

    def __init__(
        self, list_IDs, df, max_drug_nodes=290, protein_encoder=integer_label_protein
    ):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        
        self.prot_encode = protein_encoder

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]["SMILES"]
        # todo
        v_smiles = smiles2onehot(v_d)
        # todo
        v_d = self.fc(
            smiles=v_d,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )
        # Graph
        actual_node_feats = v_d.ndata.pop("h")
        num_actual_nodes = actual_node_feats.shape[0]
        if num_actual_nodes < self.max_drug_nodes:
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes

            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata["h"] = actual_node_feats
            virtual_node_feat = torch.cat(
                (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)),
                1,
            )
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})

        v_d = v_d.add_self_loop()

        # Protein seq
        v_p = self.df.iloc[index]["Protein"]
        # pdb.set_trace()
        v_p = self.prot_encode(v_p)
        # pdb.set_trace()
        y = self.df.iloc[index]["Y"]

        return v_smiles, v_d, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


#
# if __name__ == '__main__':
#     atom_featurizer = CanonicalAtomFeaturizer()
#     bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
#     fc = partial(smiles_to_bigraph, add_self_loop=True)
#     smiles = 'OC1=NN=C(CC2=CC(C(=O)N3CCN(CC3)C(=O)C3CC3)=C(F)C=C2)C2=CC=CC=C12'
#     v_d = fc(smiles=smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
#     print(v_d)
#     v = v_d.ndata.pop('h')
#
#     print(v.shape[1])
#     print(v.shape)
