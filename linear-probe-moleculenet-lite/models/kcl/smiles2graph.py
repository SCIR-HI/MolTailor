'''
Author: simwit 517992857@qq.com
Date: 2023-08-10 20:00:52
LastEditors: simwit 517992857@qq.com
LastEditTime: 2023-08-10 20:01:58
FilePath: /workspace/01-st/finetune-moleculenet/models/kcl/smiles2graph.py
Description: Source: https://github.com/ZJU-Fangyin/KCL/blob/main/code/data/smiles_to_dglgraph.py#L20

Copyright (c) 2023 by simwit, All Rights Reserved. 
'''
from rdkit import RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
import dgl
import torch
from collections import defaultdict as ddict

import sys
from pathlib import Path, PosixPath


base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/KCL'

class Triples:
    def __init__(self, data_dir = str(model_path / "triples")):
        self.data = self.load_data(data_dir)
        self.entities, self.entity2id = self.get_entities(self.data)
        self.attributes, self.attribute2id = self.get_attributes(self.data)
        self.relations, self.relation2id = self.get_relations(self.data)
        self.triples = self.read_triple(self.data, self.entity2id, self.relation2id)
        self.h2rt = self.h2rt(self.triples)
        self.t2rh = self.t2rh(self.triples)

    def load_data(self, data_dir):
        with open("%s.txt" % (data_dir), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        relationid = [i for i in range(len(relations))]
        relation2id = dict(zip(relations, relationid))
        return relations, relation2id

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        entityid = [i for i in range(len(entities))]
        entity2id = dict(zip(entities, entityid))
        return entities, entity2id

    def get_attributes(self, data):
        attributes = sorted(list(set([d[0] for d in data])))
        attributeid = [i for i in range(len(attributes))]
        attribute2id = dict(zip(attributes, attributeid))
        return attributes, attribute2id

    def read_triple(self, data, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        for triple in data:
            h = triple[0]
            r = triple[1]
            t = triple[2]
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples
        
    def h2rt(self, triples):    # dict: attribute_id  --> list[(rel_id, atom_id)]
        h2rt = ddict(list)
        for tri in triples:
            h, r, t = tri
            h2rt[h].append((r,t))
        return h2rt

    def t2rh(self, triples):    # dict: atom_id  --> list[(rel_id, attribute_id)]
        t2rh = ddict(list)
        for tri in triples:
            h, r, t = tri
            t2rh[t].append((r,h))
        return t2rh


def bondtype_features(bond):
    bondtype_list=['SINGLE','DOUBLE','TRIPLE','AROMATIC']
    bond2emb = {}
    for idx, bt in enumerate(bondtype_list):
        bond2emb[bt]= idx
    fbond = bond2emb[str(bond.GetBondType())]
    return fbond

def smiles_2_kgdgl(smiles): # augmented graph
    data = Triples()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Invalid mol found')
        return None

    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())
    
    connected_atom_list = sorted(list(set(connected_atom_list)))
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]

    # get all node ids and relations
    begin_attributes = []    # attributes
    end_atoms = []   # atoms
    rel_features = []   # relations between attributes and atoms
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx()
        symbol = atom.GetSymbol()
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue

        atoms_feature[connected_atom_map[node_index]] = atomicnum # atom nodes indexed by atomicnum

        if symbol in data.entities:
            attribute_id = [h for (r,h) in data.t2rh[data.entity2id[symbol]]] 
            rid = [r for (r,h) in data.t2rh[data.entity2id[symbol]]] # relation ids 

            begin_attributes.extend(attribute_id)   # add attribute ids
            end_atoms.extend([node_index]*len(attribute_id))    # add atom ids
            rel_features.extend(i+4 for i in rid)  # first 4 ids are prepared for bonds, relation ids begin after bond ids

    # get list of attribute ids and features
    if begin_attributes:
        attribute_id = sorted(list(set(begin_attributes)))
        node_id = [i+len(connected_atom_list) for i in range(len(attribute_id))]
        attrid2nodeid = dict(zip(attribute_id, node_id)) # dict: attribute_id in triples --> node_id in dglgraph
        nodeids = [attrid2nodeid[i] for i in begin_attributes] # list of attribute ids

        nodes_feature = [i+118 for i in attribute_id] # first 118 ids are prepared for atoms, attribute ids begin after atom ids

    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    edge_type = []
    
    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)

        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
    edge_type.extend([0]*len(bonds_feature))

    # add ids and features of attributes and relations
    if end_atoms:
        begin_indexes.extend(nodeids) 
        end_indexes.extend(end_atoms)
        atoms_feature.extend(nodes_feature)
        bonds_feature.extend(rel_features)
        edge_type.extend([1]*len(rel_features))

    
    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long) 
    graph.edata['etype'] = torch.tensor(edge_type, dtype=torch.long) # 0 for bonds & 1 for rels
    return graph